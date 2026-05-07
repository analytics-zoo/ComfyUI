import os
import torch
import logging
import comfy.model_management

RMSNorm = torch.nn.RMSNorm

log = logging.getLogger(__name__)

# ---------- omni_xpu_kernel accelerated RMSNorm ----------
# Control via env var: OMNI_XPU_KERNEL_DISABLE=1 to disable (default: enabled)
_omni_norm = None
_omni_disabled_by_env = os.environ.get("OMNI_XPU_KERNEL_DISABLE", "0") == "1"
_omni_rmsnorm_logged_first_use = False

if _omni_disabled_by_env:
    log.info("[omni_xpu_kernel] Disabled by environment variable OMNI_XPU_KERNEL_DISABLE=1")
else:
    try:
        from omni_xpu_kernel import norm as _omni_norm
        log.info("[omni_xpu_kernel] Loaded successfully — accelerated functional RMSNorm enabled")
    except ImportError:
        log.info("[omni_xpu_kernel] Not installed — using PyTorch native RMSNorm")


def _can_use_omni_rms(x):
    """Check if input is eligible for omni_xpu_kernel acceleration."""
    if _omni_norm is None:
        return False
    if not x.is_xpu:
        return False
    if x.ndim < 2:
        return False
    hidden_size = x.shape[-1]
    if hidden_size > 8192 or hidden_size % 32 != 0:
        return False
    return True


def rms_norm(x, weight=None, eps=1e-6):
    if _can_use_omni_rms(x):
        global _omni_rmsnorm_logged_first_use
        if not _omni_rmsnorm_logged_first_use:
            _omni_rmsnorm_logged_first_use = True
            log.info("[omni_xpu_kernel] First use in rmsnorm: functional rms_norm with shape %s", x.shape)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        if weight is not None:
            w = comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)
            out = _omni_norm.rms_norm(w, x_2d, eps)
        else:
            # omni rms_norm requires weight; create ones
            w = torch.ones(orig_shape[-1], dtype=x.dtype, device=x.device)
            out = _omni_norm.rms_norm(w, x_2d, eps)
        return out.reshape(orig_shape)

    # Fallback to PyTorch
    if weight is None:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
    else:
        return torch.nn.functional.rms_norm(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
