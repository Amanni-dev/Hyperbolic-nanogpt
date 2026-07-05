"""
Lorentz (hyperboloid) model operations.

All operations work on tensors whose last dimension is the ambient (d+1)
dimension.  Points on the hyperboloid H^d_k have shape (..., d+1) and satisfy
    <x, x>_L = -k  (i.e. -x0^2 + sum xi^2 = -k).

Tangent vectors also live in R^{d+1} but satisfy <x, v>_L = 0 at the base
point x.

Numerical stability is the #1 concern — every operation is clamped to avoid
NaN/Inf from:
  - arcosh(x) with x < 1  (floating point can push it below 1)
  - division by near-zero norms
  - cosh/sinh overflow when the tangent norm is large
"""

import torch
import torch.jit


@torch.jit.script
def arcosh(x: torch.Tensor):
    dtype = x.dtype
    x = x.double().clamp_min(1.0 + 1e-15)
    z = torch.sqrt(x.pow(2) - 1.0)
    return torch.log(x + z).to(dtype)


def inner(u, v, *, keepdim=False, dim=-1):
    return _inner(u, v, keepdim=keepdim, dim=dim)


@torch.jit.script
def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )


def distance(x, y, *, k, keepdim=False, dim=-1):
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    ratio = (d / k).clamp_min(1.0 + 1e-15)
    return torch.sqrt(k) * arcosh(ratio)


def project(x, *, k, dim=-1):
    return _project(x, k=k, dim=dim)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    norm_sq = (x * x).sum(dim=dim, keepdim=True)
    p0 = torch.sqrt(k + norm_sq)
    return torch.cat((p0, x), dim=dim)


def origin_like(ref, *, k, dim=-1):
    d = ref.size(dim) - 1
    o = torch.zeros_like(ref)
    o0 = torch.sqrt(k).expand_as(o.narrow(dim, 0, 1))
    o = torch.cat((o0, o.narrow(dim, 1, d)), dim=dim)
    return o


def expmap0(v, *, k, dim=-1, eps=1e-12):
    return _expmap0(v, k=k, dim=dim, eps=eps)


@torch.jit.script
def _expmap0(v, k: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    norm_sq = (v * v).sum(dim=dim, keepdim=True)
    is_zero = (norm_sq < eps * eps)
    norm = torch.sqrt(norm_sq.clamp_min(eps))
    sqrt_k = torch.sqrt(k)
    alpha = (norm / sqrt_k).clamp_max(20.0)
    p0 = sqrt_k * torch.cosh(alpha)
    pd = sqrt_k * torch.sinh(alpha) * v / norm
    pd = torch.where(is_zero, torch.zeros_like(pd), pd)
    return torch.cat((p0, pd), dim=dim)


def logmap0(x, *, k, dim=-1, eps=1e-12):
    return _logmap0(x, k=k, dim=dim, eps=eps)


@torch.jit.script
def _logmap0(x, k: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    sqrt_k = torch.sqrt(k)
    x0 = x.narrow(dim, 0, 1)
    spatial = x.narrow(dim, 1, x.size(dim) - 1)
    norm_sq = (spatial * spatial).sum(dim=dim, keepdim=True)
    is_zero = (norm_sq < eps * eps)
    norm = torch.sqrt(norm_sq.clamp_min(eps))
    ratio = (x0 / sqrt_k).clamp_min(1.0 + 1e-15)
    alpha = arcosh(ratio)
    result = sqrt_k * alpha * spatial / norm
    result = torch.where(is_zero, torch.zeros_like(result), result)
    return result


def lorentz_add(x, y, *, k, dim=-1, eps=1e-12):
    vx = _logmap0(x, k=k, dim=dim, eps=eps)
    vy = _logmap0(y, k=k, dim=dim, eps=eps)
    v = vx + vy
    return _expmap0(v, k=k, dim=dim, eps=eps)


def lorentz_scalar_mul(x, r, *, k, dim=-1, eps=1e-12):
    v = _logmap0(x, k=k, dim=dim, eps=eps)
    return _expmap0(v * r, k=k, dim=dim, eps=eps)


def weighted_midpoint(values, weights, *, k, dim=-2, eps=1e-12):
    v = _logmap0(values, k=k, dim=-1, eps=eps)
    w = weights.unsqueeze(-1)
    agg = (v * w).sum(dim=dim)
    return _expmap0(agg, k=k, dim=-1, eps=eps)


def lorentz_rms_norm(x, *, k, dim=-1, eps=1e-8):
    v = _logmap0(x, k=k, dim=dim, eps=eps)
    ms = (v * v).mean(dim=dim, keepdim=True)
    v = v * torch.rsqrt(ms + eps)
    return _expmap0(v, k=k, dim=dim, eps=eps)


def to_manifold(x_s, *, k, dim=-1):
    norm_sq = (x_s * x_s).sum(dim=dim, keepdim=True)
    x0 = torch.sqrt(k + norm_sq)
    return torch.cat((x0, x_s), dim=dim)


def space(x, *, dim=-1):
    return x.narrow(dim, 1, x.size(dim) - 1)


def sq_distance(x, y, *, k, keepdim=False, dim=-1):
    d = _dist(x, y, k=k, keepdim=keepdim, dim=dim)
    return d * d


def lorentz_centroid(points, weights, *, k, dim=-2, eps=1e-6):
    num = (points * weights.unsqueeze(-1)).sum(dim=dim)
    inn = inner(num, num, dim=-1, keepdim=True)
    denom = (-inn).clamp_min(eps).sqrt()
    return num * (torch.sqrt(k) / denom)
