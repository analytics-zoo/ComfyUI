import logging
import os
import threading
from typing import Optional

import torch


log = logging.getLogger(__name__)

_omni_linear = None
_omni_logged_first_use = False
_omni_fp8_failure_cache = set()
_omni_fp8_failure_cache_lock = threading.Lock()

try:
    from omni_xpu_kernel import linear as _omni_linear
except ImportError:
    _omni_linear = None


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _omni_fp8_enabled() -> bool:
    return _env_enabled("COMFY_XPU_FP8_OMNI_ENABLE", True)


def _omni_fp8_log_enabled() -> bool:
    return _env_enabled("COMFY_XPU_FP8_OMNI_LOG", False)


def _log_first_use(shape):
    global _omni_logged_first_use
    if not _omni_logged_first_use:
        _omni_logged_first_use = True
        log.info("[omni_xpu_kernel] First use in xpu_fp8_linear with input shape %s", shape)


def _log_fast_path_event(message: str, *args):
    if _omni_fp8_log_enabled():
        log.info(message, *args)


def _is_primitive_creation_failure(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "could not create a primitive" in message


def _primitive_failure_cache_key(input_tensor: torch.Tensor, qdata: torch.Tensor):
    return (tuple(input_tensor.shape), tuple(qdata.shape), str(input_tensor.dtype), input_tensor.device.index)


def _log_bad_shape_event(reason: str, input_tensor: torch.Tensor, qdata: torch.Tensor, bias: Optional[torch.Tensor], error: Optional[RuntimeError] = None):
    message = (
        "[omni_xpu_kernel] XPU FP8 bad shape %s input_shape=%s qdata_shape=%s dtype=%s device=%s has_bias=%s"
    )
    args = [
        reason,
        tuple(input_tensor.shape),
        tuple(qdata.shape),
        str(input_tensor.dtype),
        str(input_tensor.device),
        bias is not None,
    ]
    if error is not None:
        message += " error=%s"
        args.append(str(error))
    log.info(message, *args)


def _expand_weight_scale(scale, rows, device):
    if scale is None:
        return None
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device, dtype=torch.float32)
    scale = scale.to(device=device, dtype=torch.float32)
    if scale.ndim == 0:
        return scale.expand(rows).contiguous()
    if scale.ndim == 1 and scale.numel() == rows:
        return scale.contiguous()
    return None


def _extract_qdata(weight):
    return getattr(weight, "_qdata", None)


def _extract_params(weight):
    return getattr(weight, "_params", None)


def _extract_layout_name(weight):
    return getattr(weight, "_layout_cls", None)


def _normalize_layout_name(layout):
    if isinstance(layout, str):
        return layout
    if hasattr(layout, "__name__"):
        return layout.__name__
    if hasattr(layout, "__class__") and hasattr(layout.__class__, "__name__"):
        return layout.__class__.__name__
    return None


def can_use_omni_fp8_linear(input_tensor, weight, bias: Optional[torch.Tensor]):
    if not _omni_fp8_enabled():
        return False
    if _omni_linear is None:
        return False
    if not isinstance(input_tensor, torch.Tensor):
        return False
    if not input_tensor.is_xpu or input_tensor.ndim != 2:
        return False
    if input_tensor.dtype not in (torch.float16, torch.bfloat16):
        return False

    layout_name = _normalize_layout_name(_extract_layout_name(weight))
    if layout_name not in ("TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout"):
        return False

    qdata = _extract_qdata(weight)
    params = _extract_params(weight)
    if qdata is None or params is None:
        return False
    if not isinstance(qdata, torch.Tensor) or not qdata.is_xpu or qdata.ndim != 2:
        return False
    if qdata.device != input_tensor.device:
        return False
    if qdata.dtype != torch.float8_e4m3fn:
        return False

    scales = _expand_weight_scale(getattr(params, "scale", None), qdata.shape[0], qdata.device)
    if scales is None:
        return False

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            return False
        if not bias.is_xpu or bias.ndim != 1:
            return False
        if bias.device != input_tensor.device:
            return False
        if bias.shape[0] != qdata.shape[0]:
            return False
        if bias.dtype != input_tensor.dtype:
            return False

    if qdata.shape[1] != input_tensor.shape[1]:
        return False

    return True


def try_omni_fp8_linear(input_tensor, weight, bias: Optional[torch.Tensor]):
    if not _omni_fp8_enabled():
        _log_fast_path_event("[omni_xpu_kernel] XPU FP8 fast path disabled by COMFY_XPU_FP8_OMNI_ENABLE")
        return None

    if not can_use_omni_fp8_linear(input_tensor, weight, bias):
        _log_fast_path_event("[omni_xpu_kernel] XPU FP8 fast path fallback for shape=%s", tuple(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else None)
        return None

    qdata = _extract_qdata(weight)
    params = _extract_params(weight)
    scales = _expand_weight_scale(params.scale, qdata.shape[0], qdata.device)
    failure_key = _primitive_failure_cache_key(input_tensor, qdata)

    with _omni_fp8_failure_cache_lock:
        if failure_key in _omni_fp8_failure_cache:
            _log_bad_shape_event("cached primitive creation failure", input_tensor, qdata, bias)
            return None

    _log_first_use(tuple(input_tensor.shape))

    try:
        output = _omni_linear.onednn_w8a16_fp8(input_tensor.contiguous(), qdata.contiguous(), scales, bias=bias)
    except RuntimeError as error:
        if not _is_primitive_creation_failure(error):
            raise
        with _omni_fp8_failure_cache_lock:
            _omni_fp8_failure_cache.add(failure_key)
        _log_bad_shape_event("primitive creation failure", input_tensor, qdata, bias, error)
        return None

    _log_fast_path_event(
        "[omni_xpu_kernel] XPU FP8 fast path hit shape=%s dtype=%s cache=%s",
        tuple(input_tensor.shape),
        str(input_tensor.dtype),
        _omni_linear.fp8_cache_stats() if hasattr(_omni_linear, "fp8_cache_stats") else None,
    )
    return output
