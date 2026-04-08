import os
import sys
import types
import unittest
from typing import cast
from unittest import mock

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


class FakeQuantizedTensor:
    def __init__(self, qdata, layout_cls, scale, orig_dtype):
        self._qdata = qdata
        self._layout_cls = layout_cls
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=qdata.device, dtype=torch.float32)
        self._params = types.SimpleNamespace(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=tuple(qdata.shape),
        )

    def dequantize(self):
        scale = self._params.scale
        if scale.ndim == 0:
            scale_for_mul = scale
        elif self._qdata.ndim == 2 and scale.numel() == self._qdata.shape[0]:
            scale_for_mul = scale.unsqueeze(1)
        else:
            scale_for_mul = scale
        return self._qdata.to(self._params.orig_dtype) * scale_for_mul.to(self._params.orig_dtype)


class TestXpuFp8Linear(unittest.TestCase):
    def test_quant_ops_auto_imports_xpu_quant_layout_ops(self):
        sys.modules.pop("comfy.quant_ops", None)
        sys.modules.pop("comfy.xpu_quant_layout_ops", None)

        __import__("comfy.quant_ops")

        self.assertIn("comfy.xpu_quant_layout_ops", sys.modules)

    def test_registration_module_imports_without_comfy_kitchen_backend(self):
        module = __import__("comfy.xpu_quant_layout_ops", fromlist=["REGISTRATION_STATUS"])
        self.assertTrue(module.REGISTRATION_STATUS["attempted"])
        self.assertTrue(module.REGISTRATION_STATUS["registered"])
        self.assertIsNone(module.REGISTRATION_STATUS["error"])

    def test_fallback_linear_casts_dequantized_weight_and_bias_to_input_dtype(self):
        from comfy.xpu_quant_layout_ops import _fallback_linear

        input_tensor = torch.randn(2, 4, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=torch.tensor(2.0, dtype=torch.float32),
            orig_dtype=torch.float32,
        )
        bias = torch.randn(3, dtype=torch.float32)

        captured = {}

        def fake_linear(input_arg, weight_arg, bias_arg):
            captured["input_dtype"] = input_arg.dtype
            captured["weight_dtype"] = weight_arg.dtype
            captured["bias_dtype"] = bias_arg.dtype if bias_arg is not None else None
            return torch.zeros(2, 3, dtype=input_arg.dtype)

        with mock.patch("torch.nn.functional.linear", side_effect=fake_linear):
            output = _fallback_linear(input_tensor, weight, bias)

        self.assertEqual(captured["input_dtype"], torch.bfloat16)
        self.assertEqual(captured["weight_dtype"], torch.bfloat16)
        self.assertEqual(captured["bias_dtype"], torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_try_omni_fp8_linear_rejects_nd_input_without_adapter(self):
        from comfy import xpu_fp8_linear

        device = torch.device("cpu")
        input_tensor = torch.randn(2, 5, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNone(output)

    def test_xpu_fp8_linear_handler_flattens_nd_input_for_fast_path_and_restores_shape(self):
        from comfy.xpu_quant_layout_ops import _xpu_fp8_linear_handler

        input_tensor = torch.randn(2, 5, 4, dtype=torch.bfloat16)
        weight = object()
        bias = torch.randn(3, dtype=torch.bfloat16)
        captured = {}

        def fake_try_omni_fp8_linear(input_arg, weight_arg, bias_arg):
            captured["shape"] = tuple(input_arg.shape)
            captured["weight"] = weight_arg
            captured["bias"] = bias_arg
            return torch.arange(30, dtype=input_arg.dtype).reshape(10, 3)

        with mock.patch("comfy.xpu_quant_layout_ops.try_omni_fp8_linear", side_effect=fake_try_omni_fp8_linear):
            output = _xpu_fp8_linear_handler(None, (input_tensor, weight, bias), None)

        self.assertEqual(captured["shape"], (10, 4))
        self.assertIs(captured["weight"], weight)
        self.assertIs(captured["bias"], bias)
        self.assertEqual(tuple(output.shape), (2, 5, 3))

    def test_xpu_fp8_linear_handler_preserves_nd_shape_on_fallback(self):
        from comfy.xpu_quant_layout_ops import _xpu_fp8_linear_handler

        input_tensor = torch.randn(2, 5, 4, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=torch.tensor(2.0, dtype=torch.float32),
            orig_dtype=torch.float32,
        )
        bias = torch.randn(3, dtype=torch.float32)

        with mock.patch("comfy.xpu_quant_layout_ops.try_omni_fp8_linear", return_value=None):
            output = _xpu_fp8_linear_handler(None, (input_tensor, weight, bias), None)

        self.assertEqual(tuple(output.shape), (2, 5, 3))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_try_omni_fp8_linear_rejects_non_per_channel_scale(self):
        from comfy import xpu_fp8_linear

        device = torch.device("cpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        bad_scale = torch.ones(2, device=device, dtype=torch.float32)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=bad_scale,
            orig_dtype=torch.bfloat16,
        )

        output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)
        self.assertIsNone(output)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_falls_back_on_primitive_creation_failure(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = RuntimeError("could not create a primitive")

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNone(output)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_reraises_unrelated_runtime_errors(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = RuntimeError("some other runtime error")

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                with self.assertRaisesRegex(RuntimeError, "some other runtime error"):
                    xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_caches_primitive_creation_failure_by_shape(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = RuntimeError("could not create a primitive")

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                output_first = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)
                output_second = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNone(output_first)
        self.assertIsNone(output_second)
        self.assertEqual(fake_module.onednn_w8a16_fp8.call_count, 1)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_failure_cache_is_shape_specific(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_a = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        input_b = torch.randn(3, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = RuntimeError("could not create a primitive")

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                output_a = xpu_fp8_linear.try_omni_fp8_linear(input_a, weight, None)
                output_b = xpu_fp8_linear.try_omni_fp8_linear(input_b, weight, None)

        self.assertIsNone(output_a)
        self.assertIsNone(output_b)
        self.assertEqual(fake_module.onednn_w8a16_fp8.call_count, 2)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_logs_bad_shape_on_primitive_creation_failure(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = RuntimeError("could not create a primitive")

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                with self.assertLogs(xpu_fp8_linear.log, level="INFO") as logs:
                    output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNone(output)
        joined = "\n".join(logs.output)
        self.assertIn("primitive creation failure", joined)
        self.assertIn("input_shape=(2, 4)", joined)
        self.assertIn("qdata_shape=(3, 4)", joined)
        self.assertIn("dtype=torch.bfloat16", joined)
        self.assertIn("has_bias=False", joined)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_logs_cached_bad_shape_skip(self):
        """Cached primitive creation failure logs only at verbose level (COMFY_XPU_FP8_OMNI_LOG=2)."""
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        failure_key = xpu_fp8_linear._primitive_failure_cache_key(input_tensor, qweight)
        fake_module = mock.Mock()

        old_val = os.environ.get("COMFY_XPU_FP8_OMNI_LOG")
        os.environ["COMFY_XPU_FP8_OMNI_LOG"] = "2"
        try:
            with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
                with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", {failure_key}):
                    with self.assertLogs(xpu_fp8_linear.log, level="INFO") as logs:
                        output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)
        finally:
            if old_val is None:
                os.environ.pop("COMFY_XPU_FP8_OMNI_LOG", None)
            else:
                os.environ["COMFY_XPU_FP8_OMNI_LOG"] = old_val

        self.assertIsNone(output)
        fake_module.onednn_w8a16_fp8.assert_not_called()
        joined = "\n".join(logs.output)
        self.assertIn("cached primitive creation failure", joined)
        self.assertIn("input_shape=(2, 4)", joined)
        self.assertIn("qdata_shape=(3, 4)", joined)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_cached_failure_silent_at_default_log_level(self):
        """At default log level (1), cached primitive creation failures produce no log output."""
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        failure_key = xpu_fp8_linear._primitive_failure_cache_key(input_tensor, qweight)
        fake_module = mock.Mock()

        old_val = os.environ.get("COMFY_XPU_FP8_OMNI_LOG")
        os.environ.pop("COMFY_XPU_FP8_OMNI_LOG", None)
        try:
            with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
                with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", {failure_key}):
                    output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)
        finally:
            if old_val is not None:
                os.environ["COMFY_XPU_FP8_OMNI_LOG"] = old_val

        self.assertIsNone(output)
        fake_module.onednn_w8a16_fp8.assert_not_called()

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_accepts_layout_class_object(self):
        from comfy import xpu_fp8_linear
        from comfy.quant_ops import TensorCoreFP8E4M3Layout

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls=TensorCoreFP8E4M3Layout,
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        fake_linear = mock.Mock()
        fake_linear.onednn_w8a16_fp8.return_value = torch.ones(2, 3, device=device, dtype=torch.bfloat16)
        fake_linear.fp8_cache_stats.return_value = {"hits": 1, "misses": 1, "size": 1}

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_linear):
            output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNotNone(output)
        output = cast(torch.Tensor, output)
        self.assertEqual(tuple(output.shape), (2, 3))

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_rejects_mixed_xpu_devices(self):
        from comfy import xpu_fp8_linear

        if getattr(torch.xpu, "device_count", lambda: 0)() < 2:
            self.skipTest("Need 2 XPU devices for mixed-device rejection test")

        input_device = torch.device("xpu:0")
        weight_device = torch.device("xpu:1")
        input_tensor = torch.randn(2, 4, device=input_device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=weight_device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=torch.tensor(2.0, device=weight_device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)
        self.assertIsNone(output)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_try_omni_fp8_linear_expands_scalar_weight_scale(self):
        from comfy import xpu_fp8_linear

        device = torch.device("xpu")
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
        qweight = torch.randn(3, 4, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = FakeQuantizedTensor(
            qdata=qweight,
            layout_cls="TensorCoreFP8E4M3Layout",
            scale=torch.tensor(2.0, device=device, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
        )

        captured = {}

        def fake_linear(x, w, scales, bias=None):
            captured["scales"] = scales
            return torch.zeros(x.shape[0], w.shape[0], device=x.device, dtype=x.dtype)

        fake_module = mock.Mock()
        fake_module.onednn_w8a16_fp8.side_effect = fake_linear
        fake_module.fp8_cache_stats.return_value = {"hits": 0, "misses": 1, "size": 1}

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_module):
            with mock.patch.object(xpu_fp8_linear, "_omni_fp8_failure_cache", set()):
                output = xpu_fp8_linear.try_omni_fp8_linear(input_tensor, weight, None)

        self.assertIsNotNone(output)
        output = cast(torch.Tensor, output)
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertEqual(captured["scales"].shape[0], 3)
        self.assertTrue(torch.allclose(captured["scales"], torch.full((3,), 2.0, device=device)))


if __name__ == "__main__":
    unittest.main()
