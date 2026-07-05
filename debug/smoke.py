"""Fast correctness smoke test for the fully-hyperbolic GPT (1 fwd/bwd)."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model.model import GPT
from model.config import Config
from utils.lmath import inner

torch.manual_seed(0)
cfg = Config(data_path="data/shakespeare_char")
cfg.n_layers, cfg.n_heads, cfg.head_dim = 3, 4, 16
cfg.sequence_length = 32
cfg.n_embd = cfg.n_heads * cfg.head_dim
cfg.vocab_size = 72
cfg.curvature = 1.0

m = GPT(cfg)
print("params:", sum(p.numel() for p in m.parameters()))
idx = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_length))
tgt = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_length))

logits, loss = m(idx, targets=tgt)
print("logits", tuple(logits.shape), "loss", float(loss), "finite:", torch.isfinite(loss).item())

# manifold constraint on the final representation
k = m.k
x = m._embed(idx)
for b in m.transformer.h:
    x = b(x, k=k)
x = m.norm_f(x, k=k)
cc = inner(x, x, dim=-1)
print(f"<x,x>_L should be -{k.item():.1f}: mean={cc.mean():.5f} maxdev={(cc+k).abs().max():.2e}")
print("x0>0 everywhere:", bool((x[..., 0] > 0).all()))

loss.backward()
bad = [n for n, p in m.named_parameters() if p.grad is None]
nan = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print("NO-GRAD params:", bad)
print("NaN-grad params:", nan)
print("OK" if not bad and not nan and torch.isfinite(loss) else "PROBLEM")
