import math
import torch
from torch import nn
import torch.nn.functional as F

from utils.lmath import to_manifold, space


class LorentzEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, idx, k):
        v = F.embedding(idx, self.weight)
        return to_manifold(v, k=k, dim=-1)


class LorentzLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None,
                 zero_init=False, init_std=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(out_features, in_features + 1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters(zero_init, init_std)

    def reset_parameters(self, zero_init, init_std):
        if zero_init:
            nn.init.zeros_(self.weight)
        elif init_std is not None:
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
        else:
            std = 1.0 / math.sqrt(self.in_features + 1)
            nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, k):
        y_s = F.linear(x, self.weight, self.bias)
        if self.activation is not None:
            y_s = self.activation(y_s)
        return to_manifold(y_s, k=k, dim=-1)


class LorentzRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x, k):
        x_s = space(x, dim=-1)
        ms = (x_s * x_s).mean(dim=-1, keepdim=True)
        x_s = x_s * torch.rsqrt(ms + self.eps) * self.gain
        return to_manifold(x_s, k=k, dim=-1)


class LorentzMLR(nn.Module):
    def __init__(self, num_classes, n_embd, init_std=None):
        super().__init__()
        std = init_std if init_std is not None else 1.0 / math.sqrt(n_embd)
        self.z = nn.Parameter(torch.randn(num_classes, n_embd) * std)
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x, k):
        x0 = x.narrow(-1, 0, 1)
        xs = x.narrow(-1, 1, x.size(-1) - 1)
        znorm = self.z.norm(dim=-1).clamp_min(1e-6)
        w_time = torch.sinh(self.a) * znorm
        xz = xs @ self.z.t()
        alpha = -x0 * w_time + torch.cosh(self.a) * xz
        arg = alpha / (torch.sqrt(k) * znorm)
        return torch.exp(self.log_tau) * znorm * torch.asinh(arg)
