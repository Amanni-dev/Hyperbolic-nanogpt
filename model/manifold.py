#!/usr/bin/env python3

from abc import abstractmethod
from torch.nn import Embedding


class Manifold:
    def allocate_lt(self, N, dim, sparse):
        return Embedding(N, dim, sparse=sparse)

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    def norm(self, u, **kwargs):
        if isinstance(u, Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()
