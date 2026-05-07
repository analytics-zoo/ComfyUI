import torch
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
import logging

# ---------- omni_xpu_kernel accelerated RoPE ----------
_omni_rotary = None
_omni_rotary_logged = False

try:
    from omni_xpu_kernel import rotary as _omni_rotary
    logging.info("[omni_xpu_kernel] Loaded rotary — accelerated RoPE enabled")
except ImportError:
    pass


def _can_use_omni_rope(x):
    if _omni_rotary is None:
        return False
    if not x.is_xpu:
        return False
    head_dim = x.shape[-1]
    if head_dim not in (64, 128):
        return False
    return True


def _omni_apply_rope1(x: Tensor, freqs_cis: Tensor):
    """Apply RoPE using omni_xpu_kernel ESIMD rotary kernel."""
    global _omni_rotary_logged

    # x: [B, H, S, D], freqs_cis: [B, 1, S_freq, D/2, 2, 2]
    B, H, S, D = x.shape
    S_freq = freqs_cis.shape[2]

    # Fallback to vanilla implementation if freqs seq_len < x seq_len
    if S_freq < S:
        x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
            freqs_cis = freqs_cis[:, :, :x_.shape[2]]
        x_out = freqs_cis[..., 0] * x_[..., 0]
        x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
        return x_out.reshape(*x.shape).type_as(x)

    if not _omni_rotary_logged:
        _omni_rotary_logged = True
        logging.info("[omni_xpu_kernel] First use of ESIMD rotary_emb with shape %s", x.shape)

    # Extract cos/sin from rotation matrix pe, truncated to actual seq_len S
    # pe[..., 0, 0] = cos, pe[..., 1, 0] = sin
    cos_cache = freqs_cis[0, 0, :S, :, 0, 0].to(dtype=torch.float32).contiguous()  # [S, D/2]
    sin_cache = freqs_cis[0, 0, :S, :, 1, 0].to(dtype=torch.float32).contiguous()  # [S, D/2]

    # Reshape: [B, H, S, D] -> [B, S, H, D] -> [B*S*H, D]
    x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(B * S * H, D)

    out = _omni_rotary.rotary_emb(x_flat, cos_cache, sin_cache, S, H)

    # Reshape back: [B*S*H, D] -> [B, S, H, D] -> [B, H, S, D]
    return out.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, transformer_options={}) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)
    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if not comfy.model_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x: Tensor, freqs_cis: Tensor):
    if _can_use_omni_rope(x) and x.ndim == 4 and freqs_cis.ndim == 6:
        return _omni_apply_rope1(x, freqs_cis)

    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)


def _apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


try:
    import comfy.quant_ops
    q_apply_rope = comfy.quant_ops.ck.apply_rope
    q_apply_rope1 = comfy.quant_ops.ck.apply_rope1
    def apply_rope(xq, xk, freqs_cis):
        if comfy.model_management.in_training:
            return _apply_rope(xq, xk, freqs_cis)
        else:
            return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)
    def apply_rope1(x, freqs_cis):
        if comfy.model_management.in_training:
            return _apply_rope1(x, freqs_cis)
        if _can_use_omni_rope(x) and x.ndim == 4 and freqs_cis.ndim == 6:
            return _omni_apply_rope1(x, freqs_cis)
        return q_apply_rope1(x, freqs_cis)
except:
    logging.warning("No comfy kitchen, using old apply_rope functions.")
    apply_rope = _apply_rope
    apply_rope1 = _apply_rope1
