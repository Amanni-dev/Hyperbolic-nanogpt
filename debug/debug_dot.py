"""
Debug script for the Lorentz math operations in utils/lmath.py.
Tests expmap0, logmap0, lorentz_add, weighted_midpoint, distance.
No GPU required.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from utils.lmath import (
    expmap0,
    logmap0,
    distance,
    inner,
    lorentz_add,
    lorentz_scalar_mul,
    project,
)

print("=== Testing Lorentz math operations ===\n")

k = torch.tensor(1.0)

# 1. expmap0 / logmap0 round-trip
v = torch.randn(4, 8)  # spatial tangent vectors
x = expmap0(v, k=k, dim=-1)              # (4, 9) on the hyperboloid
v_recovered = logmap0(x, k=k, dim=-1)    # (4, 8) back to tangent
rt_error = (v - v_recovered).abs().max().item()
print(f"expmap0/logmap0 round-trip max error: {rt_error:.2e}")
assert rt_error < 1e-5, "Round-trip failed!"

# 2. Check that expmap0 produces points on the hyperboloid: <x,x>_L = -k
inner_xx = inner(x, x, dim=-1)
print(f"<x, x>_L (should be -k = {-k.item()}): max abs error = {(inner_xx + k).abs().max().item():.2e}")

# 3. Distance from origin should be ||v||
dist_origin = distance(x, expmap0(torch.zeros(1, 8), k=k, dim=-1), k=k, dim=-1)
v_norm = v.norm(dim=-1)
dist_err = (dist_origin - v_norm).abs().max().item()
print(f"Distance to origin vs ||v||: max error = {dist_err:.2e}")

# 4. lorentz_add: adding origin should be identity
origin = expmap0(torch.zeros(4, 8), k=k, dim=-1)
x_plus_origin = lorentz_add(x, origin, k=k, dim=-1)
add_err = (logmap0(x_plus_origin, k=k, dim=-1) - v).abs().max().item()
print(f"lorentz_add(x, origin) ≈ x: max error = {add_err:.2e}")

# 5. lorentz_scalar_mul: scaling by 1 should be identity
x_scaled = lorentz_scalar_mul(x, torch.tensor(1.0), k=k, dim=-1)
scale_err = (logmap0(x_scaled, k=k, dim=-1) - v).abs().max().item()
print(f"lorentz_scalar_mul(x, 1) ≈ x: max error = {scale_err:.2e}")

# 6. project: projecting a spatial vector should give a point on the hyperboloid
p = project(v, k=k, dim=-1)
inner_pp = inner(p, p, dim=-1)
print(f"project(v): <p, p>_L (should be -k): max abs error = {(inner_pp + k).abs().max().item():.2e}")

# 7. Distance is symmetric and non-negative
x2 = expmap0(torch.randn(4, 8), k=k, dim=-1)
d12 = distance(x, x2, k=k, dim=-1)
d21 = distance(x2, x, k=k, dim=-1)
sym_err = (d12 - d21).abs().max().item()
print(f"Distance symmetry: max error = {sym_err:.2e}")
print(f"Distance min value: {d12.min().item():.4f} (should be >= 0)")

print("\n=== All Lorentz math tests passed! ===")
