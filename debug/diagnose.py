"""
Overfit validation for the FULLY-HYPERBOLIC GPT.

Decisive test: can the model overfit a single fixed batch of real Shakespeare?
A model that actually routes context must drive train loss WELL below the
bigram floor (~2.49 nats for this corpus).  The old (tangent-collapsed) model
plateaued exactly at the bigram floor.

Also tracks: attention entropy (should DROP as it learns to route), spatial
norm of activations (stability), and NaN/Inf.
"""

import sys
import math
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch
import torch.nn.functional as F

from custom_tokenizers.char_tokenizer import CharacterTokenizer
from model.model import GPT, HypSelfAttention
from model.config import Config
from utils.lmath import to_manifold, space, sq_distance, lorentz_centroid, inner

torch.manual_seed(0)
np.random.seed(0)

_ATTN = []


def _capturing_forward(self, x, k):
    B, T, _ = x.size()
    C, nh, hd = self.n_embd, self.n_heads, self.head_dim
    qkv = self.c_attn(x, k=k)
    s = space(qkv, dim=-1)
    q_s, k_s, v_s = s.split(C, dim=-1)

    def heads(t):
        return t.view(B, T, nh, hd).transpose(1, 2)

    q = to_manifold(heads(q_s), k=k, dim=-1)
    kk = to_manifold(heads(k_s), k=k, dim=-1)
    v = to_manifold(heads(v_s), k=k, dim=-1)
    d2 = sq_distance(q.unsqueeze(-2), kk.unsqueeze(-3), k=k, dim=-1)
    scores = -d2 * torch.exp(self.log_scale)
    scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    wei = F.softmax(scores, dim=-1)
    _ATTN.append(wei.detach())
    out = lorentz_centroid(v.unsqueeze(2), wei, k=k, dim=-2)
    out_s = space(out, dim=-1).transpose(1, 2).contiguous().view(B, T, C)
    y = to_manifold(out_s, k=k, dim=-1)
    return self.c_proj(y, k=k)


def attn_rel_entropy(wei_list):
    ents = []
    for wei in wei_list:
        B, nh, Tq, Tk = wei.shape
        w = wei.clamp_min(1e-12)
        row_ent = -(w * w.log()).sum(-1)                 # (B,nh,Tq)
        idx = torch.arange(1, Tq + 1, dtype=w.dtype).clamp_min(1)
        norm = torch.log(idx).clamp_min(1e-6)
        rel = (row_ent / norm.view(1, 1, -1))[:, :, 1:]  # skip i=0
        ents.append(rel.mean().item())
    return float(np.mean(ents)) if ents else float('nan')


def main():
    tok = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")
    cfg = Config(data_path="data/shakespeare_char")
    cfg.n_layers, cfg.n_heads, cfg.head_dim = 2, 2, 8
    cfg.sequence_length = 16
    cfg.n_embd = cfg.n_heads * cfg.head_dim
    cfg.curvature = 1.0
    cfg.vocab_size = tok.vocab_size

    HypSelfAttention.forward = _capturing_forward

    model = GPT(cfg)
    print(f"model {model.model_size()}  vocab={cfg.vocab_size}")

    with open("data/shakespeare_char/train.bin", "rb") as f:
        f.read(256 * 4)
        data = torch.tensor(np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64))
    B, T = 8, cfg.sequence_length
    ix = torch.randint(0, len(data) - T - 1, (B,))
    batch = torch.stack([data[i:i + T + 1] for i in ix])
    X, Y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

    cnt = torch.bincount(Y.reshape(-1), minlength=cfg.vocab_size).float()
    p = cnt / cnt.sum()
    unigram = -(p[p > 0] * p[p > 0].log()).sum().item()
    print(f"batch unigram floor = {unigram:.3f} nats | corpus bigram floor = 2.49 nats")
    print(f"target: overfit should push loss FAR below these.\n")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-9)

    print(f"{'step':>5} {'loss':>8} {'attn_ent':>9} {'xs_norm':>8} {'nan':>4}")
    STEPS = 150
    for s in range(STEPS + 1):
        _ATTN.clear()
        logits, loss = model(X, targets=Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 15 == 0:
            with torch.no_grad():
                x = model._embed(X)
                for blk in model.transformer.h:
                    x = blk(x, k=model.k)
                xs_norm = space(x, dim=-1).norm(dim=-1).mean().item()
            ent = attn_rel_entropy(_ATTN)
            nan = not torch.isfinite(loss).item()
            print(f"{s:>5} {loss.item():>8.4f} {ent:>9.3f} {xs_norm:>8.2f} "
                  f"{str(nan):>4}")

    final = loss.item()
    print(f"\nfinal overfit loss = {final:.4f}")
    if final < 1.0:
        print(">>> PASS: model routes context (overfits far below bigram 2.49).")
    elif final < 2.2:
        print(">>> PARTIAL: below bigram floor -> context IS used; tune capacity/steps.")
    else:
        print(">>> FAIL: still stuck at/above bigram floor.")


if __name__ == "__main__":
    main()
