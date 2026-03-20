import unittest
import torch
import sys
import os
import json
from unittest import mock

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor
import comfy.utils


class SimpleModel(torch.nn.Module):
    def __init__(self, operations=ops.disable_weight_init, device="cpu"):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device=device, dtype=torch.bfloat16)
        self.layer2 = operations.Linear(20, 30, device=device, dtype=torch.bfloat16)
        self.layer3 = operations.Linear(30, 40, device=device, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        return x


class TestMixedPrecisionOps(unittest.TestCase):

    def test_all_layers_standard(self):
        """Test that model with no quantization works normally"""
        # Create model
        model = SimpleModel(operations=ops.mixed_precision_ops({}))

        # Initialize weights manually
        model.layer1.weight = torch.nn.Parameter(torch.randn(20, 10, dtype=torch.bfloat16))
        model.layer1.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16))
        model.layer2.weight = torch.nn.Parameter(torch.randn(30, 20, dtype=torch.bfloat16))
        model.layer2.bias = torch.nn.Parameter(torch.randn(30, dtype=torch.bfloat16))
        model.layer3.weight = torch.nn.Parameter(torch.randn(40, 30, dtype=torch.bfloat16))
        model.layer3.bias = torch.nn.Parameter(torch.randn(40, dtype=torch.bfloat16))

        # Initialize weight_function and bias_function
        for layer in [model.layer1, model.layer2, model.layer3]:
            layer.weight_function = []
            layer.bias_function = []

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_mixed_precision_load(self):
        """Test loading a mixed precision model from state dict"""
        # Configure mixed precision: layer1 is FP8, layer2 and layer3 are standard
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            },
            "layer3": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create state dict with mixed precision
        fp8_weight1 = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(40, 30, dtype=torch.float32).to(torch.float8_e4m3fn)

        state_dict = {
            # Layer 1: FP8 E4M3FN
            "layer1.weight": fp8_weight1,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),

            # Layer 2: Standard BF16
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),

            # Layer 3: FP8 E4M3FN
            "layer3.weight": fp8_weight3,
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
            "layer3.weight_scale": torch.tensor(1.5, dtype=torch.float32),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        # Create model and load state dict (strict=False because custom loading pops keys)
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Verify weights are wrapped in QuantizedTensor
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Layer 2 should NOT be quantized
        self.assertNotIsInstance(model.layer2.weight, QuantizedTensor)

        # Layer 3 should be quantized
        self.assertIsInstance(model.layer3.weight, QuantizedTensor)
        self.assertEqual(model.layer3.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify scales were loaded
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
        self.assertEqual(model.layer3.weight._params.scale.item(), 1.5)

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_state_dict_quantized_preserved(self):
        """Test that quantized weights are preserved in state_dict()"""
        # Configure mixed precision
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict1 = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(3.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict1, _ = comfy.utils.convert_old_quants(state_dict1, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict1, strict=False)

        # Save state dict
        state_dict2 = model.state_dict()

        # Verify layer1.weight is a QuantizedTensor with scale preserved
        self.assertTrue(torch.equal(state_dict2["layer1.weight"].view(torch.uint8), fp8_weight.view(torch.uint8)))
        self.assertEqual(state_dict2["layer1.weight_scale"].item(), 3.0)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify non-quantized layers are standard tensors
        self.assertNotIsInstance(state_dict2["layer2.weight"], QuantizedTensor)
        self.assertNotIsInstance(state_dict2["layer3.weight"], QuantizedTensor)

    def test_weight_function_compatibility(self):
        """Test that weight_function (LoRA) works with quantized layers"""
        # Configure FP8 quantization
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Add a weight function (simulating LoRA)
        # This should trigger dequantization during forward pass
        def apply_lora(weight):
            lora_delta = torch.randn_like(weight) * 0.01
            return weight + lora_delta

        model.layer1.weight_function.append(apply_lora)

        # Forward pass should work with LoRA (triggers weight_function path)
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_mixed_precision_forward_reaches_registered_fp8_linear_handler(self):
        """Test that a real mixed_precision forward reaches the registered FP8 linear handler."""
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)

        handler_calls = []

        def fake_try_omni_fp8_linear(input_tensor_arg, weight_arg, bias_arg):
            handler_calls.append((input_tensor_arg, weight_arg, bias_arg))
            return None

        with mock.patch("comfy.xpu_quant_layout_ops.try_omni_fp8_linear", side_effect=fake_try_omni_fp8_linear):
            with torch.inference_mode():
                output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(len(handler_calls), 1)
        handler_input, handler_weight, handler_bias = handler_calls[0]
        self.assertEqual(tuple(handler_input.shape), (5, 10))
        self.assertIsInstance(handler_weight, QuantizedTensor)
        self.assertEqual(handler_weight._layout_cls, "TensorCoreFP8E4M3Layout")
        self.assertEqual(tuple(handler_bias.shape), (20,))

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_mixed_precision_xpu_forward_invokes_omni_kernel_fast_path(self):
        """Test that XPU mixed_precision forward hits the omni FP8 kernel fast path."""
        from comfy import xpu_fp8_linear

        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}), device="xpu")
        model.load_state_dict(state_dict, strict=False)

        input_tensor = torch.randn(5, 10, device="xpu", dtype=torch.bfloat16)
        fake_linear = mock.Mock()
        fake_linear.onednn_w8a16_fp8.return_value = torch.ones(5, 20, device="xpu", dtype=torch.bfloat16)

        with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_linear):
            with torch.inference_mode():
                output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertGreaterEqual(fake_linear.onednn_w8a16_fp8.call_count, 1)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_mixed_precision_xpu_forward_reuses_fp8_cache_for_same_shape(self):
        """Test that repeated mixed_precision XPU forwards reuse the omni FP8 cache for the same shape."""
        from omni_xpu_kernel import linear

        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}), device="xpu")
        model.load_state_dict(state_dict, strict=False)

        input_tensor = torch.randn(5, 10, device="xpu", dtype=torch.bfloat16)

        linear.fp8_cache_clear()
        self.assertEqual(linear.fp8_cache_stats(), {"hits": 0, "misses": 0, "size": 0})

        with torch.inference_mode():
            output_first = model(input_tensor)
            output_second = model(input_tensor)

        stats = linear.fp8_cache_stats()
        self.assertEqual(tuple(output_first.shape), (5, 40))
        self.assertEqual(tuple(output_second.shape), (5, 40))
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hits"], 1)
        self.assertEqual(stats["size"], 1)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_mixed_precision_xpu_forward_respects_omni_disable_env(self):
        """Test that disabling the ComfyUI omni FP8 env prevents fast-path invocation."""
        from comfy import xpu_fp8_linear

        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}), device="xpu")
        model.load_state_dict(state_dict, strict=False)

        input_tensor = torch.randn(5, 10, device="xpu", dtype=torch.bfloat16)
        fake_linear = mock.Mock()
        fake_linear.onednn_w8a16_fp8.return_value = torch.ones(5, 20, device="xpu", dtype=torch.bfloat16)

        with mock.patch.dict(os.environ, {"COMFY_XPU_FP8_OMNI_ENABLE": "0"}, clear=False):
            with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_linear):
                with torch.inference_mode():
                    output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(fake_linear.onednn_w8a16_fp8.call_count, 0)

    @unittest.skipUnless(has_xpu(), "XPU not available")
    def test_mixed_precision_xpu_forward_logs_fast_path_when_enabled(self):
        """Test that verbose log level (COMFY_XPU_FP8_OMNI_LOG=2) records fast-path hits."""
        from comfy import xpu_fp8_linear

        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}), device="xpu")
        model.load_state_dict(state_dict, strict=False)

        input_tensor = torch.randn(5, 10, device="xpu", dtype=torch.bfloat16)
        fake_linear = mock.Mock()
        fake_linear.onednn_w8a16_fp8.return_value = torch.ones(5, 20, device="xpu", dtype=torch.bfloat16)

        with mock.patch.dict(os.environ, {"COMFY_XPU_FP8_OMNI_LOG": "2"}, clear=False):
            with mock.patch.object(xpu_fp8_linear, "_omni_linear", fake_linear):
                with self.assertLogs("comfy.xpu_fp8_linear", level="INFO") as logs:
                    with torch.inference_mode():
                        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertGreaterEqual(fake_linear.onednn_w8a16_fp8.call_count, 1)
        self.assertTrue(any("fast path" in line.lower() for line in logs.output))

    def test_error_handling_unknown_format(self):
        """Test that unknown formats raise error"""
        # Configure with unknown format
        layer_quant_config = {
            "layer1": {
                "format": "unknown_format_xyz",
                "params": {}
            }
        }

        # Create state dict
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.bfloat16),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})

        # Load should raise KeyError for unknown format in QUANT_FORMAT_MIXINS
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        with self.assertRaises(KeyError):
            model.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    unittest.main()
