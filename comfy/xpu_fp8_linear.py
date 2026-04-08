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

# M-chunking: (shape_key) → chunk_m for shapes that benefit from FP8 chunking
_omni_fp8_m_chunk_cache: dict = {}
_omni_fp8_m_chunk_cache_lock = threading.Lock()

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


def _omni_fp8_log_level() -> int:
    """Return log verbosity level for FP8 operations.

    0 = off (no logging)
    1 = misses only (primitive creation failures, bad shapes — first occurrence)
    2 = verbose (all events including hits and cached failures)

    Default is 1 (misses only) when COMFY_XPU_FP8_OMNI_LOG is not set.
    Set COMFY_XPU_FP8_OMNI_LOG=0 to disable, =2 or =verbose for full logging.
    """
    value = os.environ.get("COMFY_XPU_FP8_OMNI_LOG")
    if value is None:
        return 1  # default: misses only
    value = value.strip().lower()
    if value in {"0", "false", "no", "off", ""}:
        return 0
    if value in {"2", "verbose", "debug", "all"}:
        return 2
    # "1", "true", "yes", "on" → level 1
    return 1


def _log_first_use(shape):
    global _omni_logged_first_use
    if not _omni_logged_first_use:
        _omni_logged_first_use = True
        log.info("[omni_xpu_kernel] First use in xpu_fp8_linear with input shape %s", shape)


def _log_miss_event(message: str, *args):
    """Log cache miss / first failure events (level >= 1)."""
    if _omni_fp8_log_level() >= 1:
        log.info(message, *args)


def _log_verbose_event(message: str, *args):
    """Log verbose events: hits, cached failures, fallbacks (level >= 2)."""
    if _omni_fp8_log_level() >= 2:
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

    # First failure (with error) → miss-level; cached failure → verbose-level
    if error is not None:
        _log_miss_event(message, *args)
    else:
        _log_verbose_event(message, *args)


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


# ---------------------------------------------------------------------------
# M-chunking: split large M into chunks for oneDNN FP8 primitive creation.
# oneDNN fails to JIT-compile FP8 kernels for large M; chunking works around
# this while still being faster than dequant+bf16 for FFN-shaped GEMMs.
# ---------------------------------------------------------------------------

_M_CHUNK_THRESHOLD = 4096  # below this, direct FP8 path works

# (K, N) -> chunk_m.  Only shapes where FP8 M-chunking beats dequant+bf16.
_M_CHUNK_TABLE = {
    (5120, 13824): 512,   # WAN 2.2 14B FFN up:   4% faster
    (13824, 5120): 512,   # WAN 2.2 14B FFN down: 8% faster
}


def _select_chunk_m(m: int, k: int, n: int, dtype: torch.dtype) -> Optional[int]:
    """Return optimal chunk_m for M-chunking, or None if not beneficial."""
    if m <= _M_CHUNK_THRESHOLD:
        return None
    return _M_CHUNK_TABLE.get((k, n))


def _try_fp8_m_chunked(
    input_tensor: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    chunk_m: int,
) -> Optional[torch.Tensor]:
    """Execute FP8 GEMM by splitting M dimension into chunks of size chunk_m."""
    m, k = input_tensor.shape
    n = qdata.shape[0]

    output = torch.empty(m, n, dtype=input_tensor.dtype, device=input_tensor.device)

    for start in range(0, m, chunk_m):
        end = min(start + chunk_m, m)
        x_chunk = input_tensor[start:end].contiguous()

        try:
            out_chunk = _omni_linear.onednn_w8a16_fp8(
                x_chunk, qdata, scales, bias=bias,
            )
        except RuntimeError as e:
            if _is_primitive_creation_failure(e):
                _log_miss_event(
                    "[omni_xpu_kernel] XPU FP8 M-chunking failed at chunk [%d:%d] "
                    "chunk_shape=(%d,%d) error=%s",
                    start, end, end - start, k, str(e),
                )
                return None
            raise

        output[start:end] = out_chunk

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        _log_verbose_event("[omni_xpu_kernel] XPU FP8 fast path disabled by COMFY_XPU_FP8_OMNI_ENABLE")
        return None

    if not can_use_omni_fp8_linear(input_tensor, weight, bias):
        _log_verbose_event("[omni_xpu_kernel] XPU FP8 fast path fallback for shape=%s", tuple(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else None)
        return None

    qdata = _extract_qdata(weight)
    params = _extract_params(weight)
    scales = _expand_weight_scale(params.scale, qdata.shape[0], qdata.device)
    failure_key = _primitive_failure_cache_key(input_tensor, qdata)

    with _omni_fp8_failure_cache_lock:
        if failure_key in _omni_fp8_failure_cache:
            _log_bad_shape_event("cached primitive creation failure", input_tensor, qdata, bias)
            with _omni_fp8_m_chunk_cache_lock:
                cached_chunk_m = _omni_fp8_m_chunk_cache.get(failure_key)
            if cached_chunk_m is not None:
                _log_verbose_event(
                    "[omni_xpu_kernel] XPU FP8 M-chunking (cached) chunk_m=%d for shape %s",
                    cached_chunk_m, tuple(input_tensor.shape),
                )
                return _try_fp8_m_chunked(
                    input_tensor.contiguous(), qdata.contiguous(), scales, bias,
                    cached_chunk_m,
                )
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

        # Try M-chunking for this failed shape
        m, k = input_tensor.shape
        n = qdata.shape[0]
        chunk_m = _select_chunk_m(m, k, n, input_tensor.dtype)
        if chunk_m is not None:
            result = _try_fp8_m_chunked(
                input_tensor.contiguous(), qdata.contiguous(), scales, bias,
                chunk_m,
            )
            if result is not None:
                # Cache successful chunk_m
                with _omni_fp8_m_chunk_cache_lock:
                    _omni_fp8_m_chunk_cache[failure_key] = chunk_m
                _log_miss_event(
                    "[omni_xpu_kernel] XPU FP8 M-chunking success, cached chunk_m=%d for shape %s",
                    chunk_m, tuple(input_tensor.shape),
                )
                return result

        return None

    _log_verbose_event(
        "[omni_xpu_kernel] XPU FP8 fast path hit input_shape=%s weight_shape=%s dtype=%s cache=%s",
        tuple(input_tensor.shape),
        tuple(qdata.shape),
        str(input_tensor.dtype),
        _omni_linear.fp8_cache_stats() if hasattr(_omni_linear, "fp8_cache_stats") else None,
    )
    return output
