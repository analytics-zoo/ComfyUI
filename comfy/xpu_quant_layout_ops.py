import logging

import torch

from comfy.quant_ops import register_layout_op, TensorCoreFP8E4M3Layout, TensorCoreFP8Layout
from comfy.xpu_fp8_linear import try_omni_fp8_linear


log = logging.getLogger(__name__)

REGISTRATION_STATUS = {
    "attempted": False,
    "registered": False,
    "error": None,
}


def _fallback_linear(input_tensor, weight, bias):
    target_dtype = getattr(getattr(input_tensor, "_params", None), "orig_dtype", None)
    if target_dtype is None and isinstance(input_tensor, torch.Tensor):
        target_dtype = input_tensor.dtype

    if hasattr(weight, "dequantize"):
        weight = weight.dequantize()
    if hasattr(input_tensor, "dequantize"):
        input_tensor = input_tensor.dequantize()

    if target_dtype is not None:
        if isinstance(input_tensor, torch.Tensor) and input_tensor.dtype != target_dtype:
            input_tensor = input_tensor.to(dtype=target_dtype)
        if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
            weight = weight.to(dtype=target_dtype)
        if isinstance(bias, torch.Tensor) and bias.dtype != target_dtype:
            bias = bias.to(dtype=target_dtype)

    return torch.nn.functional.linear(input_tensor, weight, bias)


def _xpu_fp8_linear_handler(func, args, kwargs):
    kwargs = kwargs or {}
    input_tensor = args[0]
    weight = args[1]
    bias = None
    if len(args) > 2:
        bias = args[2]
    elif "bias" in kwargs:
        bias = kwargs["bias"]

    reshape_back_to = None
    fast_path_input = input_tensor
    if isinstance(input_tensor, torch.Tensor) and input_tensor.ndim > 2:
        reshape_back_to = tuple(input_tensor.shape[:-1])
        fast_path_input = input_tensor.reshape(-1, input_tensor.shape[-1])

    output = try_omni_fp8_linear(fast_path_input, weight, bias)
    if output is not None:
        if reshape_back_to is not None:
            output = output.reshape(*reshape_back_to, output.shape[-1])
        return output

    return _fallback_linear(input_tensor, weight, bias)


try:
    REGISTRATION_STATUS["attempted"] = True
    register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8E4M3Layout)(_xpu_fp8_linear_handler)
    register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8Layout)(_xpu_fp8_linear_handler)
    REGISTRATION_STATUS["registered"] = True
except Exception as e:
    REGISTRATION_STATUS["error"] = str(e)
    log.info("[omni_xpu_kernel] XPU FP8 layout registration skipped: %s", e)
