"""
Fully-hyperbolic GPT on the Lorentz (hyperboloid) model H^d_k.

Every hidden state is a point on H^d_k (shape (..., d+1), <x,x>_L = -k).  All
operations act *directly* on the manifold (no log_0/exp_0 round-trip at the
origin), so composing them stays genuinely hyperbolic:

  - embeddings          : spatial coords -> hyperboloid (curvature constraint)
  - linear layers       : Lorentz-direct (rotations + boosts, Chen et al. 2022)
  - attention scores    : softmax(-d^2_L / sqrt(hd))  (HyboNet / Hypformer)
  - value aggregation   : weighted Lorentzian centroid (Law et al. 2019)
  - residual            : 2-point Lorentzian centroid
  - normalisation       : spatial RMS-norm + reprojection
  - LM head             : Lorentz MLR (signed distance to class hyperplanes)

Curvature is a FIXED constant (learnable curvature previously collapsed toward
the Euclidean limit and merely acted as a logit temperature).  Output sharpness
is handled by the explicit learnable temperature inside LorentzMLR.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from utils.lmath import (
    to_manifold,
    space,
    inner,
    sq_distance,
    lorentz_centroid,
)
from model.hyp_layers import (
    LorentzEmbedding,
    LorentzLinear,
    LorentzRMSNorm,
    LorentzMLR,
)


def _centroid2(x, sub, w_sub, k, eps=1e-6):
    num = x + w_sub * sub
    innv = inner(num, num, dim=-1, keepdim=True)
    denom = (-innv).clamp_min(eps).sqrt()
    return num * (torch.sqrt(k) / denom)


class HypSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        assert self.n_embd % self.n_heads == 0

        self.c_attn = LorentzLinear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = LorentzLinear(config.n_embd, config.n_embd, bias=True,
                                    zero_init=True)

        self.log_scale = nn.Parameter(
            torch.full((1, self.n_heads, 1, 1), -0.5 * math.log(self.head_dim))
        )

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.sequence_length, config.sequence_length)
                       ).view(1, 1, config.sequence_length, config.sequence_length)
        )

    def forward(self, x, k):
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

        out = lorentz_centroid(v.unsqueeze(2), wei, k=k, dim=-2)

        out_s = space(out, dim=-1)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, C)
        y = to_manifold(out_s, k=k, dim=-1)
        return self.c_proj(y, k=k)


class HypMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = LorentzLinear(config.n_embd, 4 * config.n_embd, bias=True,
                                  activation=F.gelu)
        self.c_proj = LorentzLinear(4 * config.n_embd, config.n_embd, bias=True,
                                    zero_init=True)

    def forward(self, x, k):
        x = self.c_fc(x, k=k)
        return self.c_proj(x, k=k)


class HypBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LorentzRMSNorm(config.n_embd)
        self.attn = HypSelfAttention(config)
        self.norm2 = LorentzRMSNorm(config.n_embd)
        self.mlp = HypMLP(config)
        self.res_attn = nn.Parameter(torch.tensor(-3.0))
        self.res_mlp = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x, k):
        a = self.attn(self.norm1(x, k=k), k=k)
        x = _centroid2(x, a, F.softplus(self.res_attn), k)
        m = self.mlp(self.norm2(x, k=k), k=k)
        x = _centroid2(x, m, F.softplus(self.res_mlp), k)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.register_buffer('k_buf', torch.tensor(float(config.curvature)))

        self.transformer = nn.ModuleDict(dict(
            wte=LorentzEmbedding(config.vocab_size, config.n_embd),
            wpe=LorentzEmbedding(config.sequence_length, config.n_embd),
            h=nn.ModuleList([HypBlock(config) for _ in range(config.n_layers)]),
        ))
        self.norm_f = LorentzRMSNorm(config.n_embd)
        self.lm_head = LorentzMLR(config.vocab_size, config.n_embd)

        self.tok_w = nn.Parameter(torch.tensor(0.0))
        self.pos_w = nn.Parameter(torch.tensor(0.0))

    @property
    def k(self):
        return self.k_buf

    def _embed(self, idx):
        k = self.k
        B, T = idx.size()
        tok = self.transformer.wte(idx, k=k)
        pos = self.transformer.wpe(
            torch.arange(0, T, dtype=torch.long, device=idx.device), k=k
        ).unsqueeze(0)
        wt = F.softplus(self.tok_w)
        wp = F.softplus(self.pos_w)
        num = wt * tok + wp * pos
        innv = inner(num, num, dim=-1, keepdim=True)
        denom = (-innv).clamp_min(1e-6).sqrt()
        return num * (torch.sqrt(k) / denom)

    def forward(self, idx, targets=None, return_logits=True):
        k = self.k
        x = self._embed(idx)
        for block in self.transformer.h:
            x = block(x, k=k)
        x = self.norm_f(x, k=k)

        if targets is not None:
            logits = self.lm_head(x, k=k).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :], k=k).float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

    @torch.no_grad()
    def generate_text(self, context, max_length=200, temperature=1.0, top_k=50):
        self.eval()
        block_size = self.config.sequence_length
        generated = context.clone()
        for _ in range(max_length):
            idx = generated if generated.size(1) <= block_size \
                else generated[:, -block_size:]
            logits, _ = self(idx, return_logits=True)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e6:
            return f"{total_params / 1e6:.3g}M"
        return f"{total_params / 1e3:.3g}K"
