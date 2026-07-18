"""Neural recommenders: NeuralCF and DeepFM.

Matrix factorization scores a (user, item) pair by a DOT product of their
latent vectors -- a fixed, linear interaction. These learn a more flexible
interaction with a neural network over the embeddings, at the cost of more data
and compute. Both take ``triples`` of ``(user, item, rating)``.
"""
import numpy as np

from ..base import BaseEstimator
from ..autograd import Tensor
from .. import nn


def _dims(triples):
    triples = [(int(u), int(i), float(r)) for u, i, r in triples]
    nu = max(u for u, _, _ in triples) + 1
    ni = max(i for _, i, _ in triples) + 1
    return triples, nu, ni


class NeuralCF(BaseEstimator):
    """Neural Collaborative Filtering (He et al., 2017).

    THE GENERALISATION OF THE DOT PRODUCT
    -------------------------------------
    Matrix factorization predicts ``user_vec . item_vec`` -- forcing the
    user-item interaction to be linear (a dot product). NeuralCF embeds the user
    and item, CONCATENATES the two vectors, and passes them through an MLP, so the
    interaction can be an arbitrary learned function. This captures nonlinear
    affinities (a user who likes A and B but not their "average") that a dot
    product cannot represent -- the reason neural CF can beat classic MF given
    enough interaction data.
    """

    def __init__(self, n_factors=16, hidden=(32, 16), epochs=30, lr=0.01,
                 batch_size=64, random_state=None):
        self.n_factors = n_factors
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, triples):
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        triples, nu, ni = _dims(triples)
        self.n_users_, self.n_items_ = nu, ni
        self.user_emb_ = nn.Embedding(nu, self.n_factors, rng=rng)
        self.item_emb_ = nn.Embedding(ni, self.n_factors, rng=rng)
        dims = [2 * self.n_factors, *self.hidden, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b, rng=rng), nn.ReLU()]
        self.mlp_ = nn.Sequential(*layers[:-1])       # drop trailing ReLU
        params = (list(self.user_emb_.parameters()) + list(self.item_emb_.parameters())
                  + list(self.mlp_.parameters()))
        opt = nn.Adam(params, lr=self.lr)
        U = np.array([u for u, _, _ in triples])
        I = np.array([i for _, i, _ in triples])
        R = np.array([r for _, _, r in triples])
        self.global_mean_ = float(R.mean())
        n = len(R)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                pred = self._forward(U[b], I[b])
                ((pred - Tensor(R[b] - self.global_mean_)) ** 2).mean().backward()
                opt.step()
        return self

    def _forward(self, u, i):
        e = Tensor.concatenate([self.user_emb_(np.asarray(u)),
                                self.item_emb_(np.asarray(i))], axis=1)
        return self.mlp_(e).reshape(-1)

    def predict(self, user, item):
        if user >= self.n_users_ or item >= self.n_items_:
            return self.global_mean_
        return float(self._forward([user], [item]).data[0]) + self.global_mean_


class DeepFM(BaseEstimator):
    """DeepFM: a factorization machine AND a deep net, sharing embeddings
    (Guo et al., 2017).

    A factorization machine captures all pairwise feature interactions cheaply
    (LOW-order); a deep net captures HIGH-order nonlinear ones. DeepFM runs BOTH on
    the SAME embeddings and sums their outputs, so it learns low- and high-order
    interactions jointly with no manual feature engineering and no separate
    pre-training. Here the "features" are the user and item ids; the FM component
    is the dot product plus biases, the deep component an MLP over the embeddings.
    """

    def __init__(self, n_factors=16, hidden=(32,), epochs=30, lr=0.01,
                 batch_size=64, random_state=None):
        self.n_factors = n_factors
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, triples):
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        triples, nu, ni = _dims(triples)
        self.n_users_, self.n_items_ = nu, ni
        self.user_emb_ = nn.Embedding(nu, self.n_factors, rng=rng)
        self.item_emb_ = nn.Embedding(ni, self.n_factors, rng=rng)
        self.ub_ = nn.Embedding(nu, 1, rng=rng)       # first-order (bias) terms
        self.ib_ = nn.Embedding(ni, 1, rng=rng)
        dims = [2 * self.n_factors, *self.hidden, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b, rng=rng), nn.ReLU()]
        self.mlp_ = nn.Sequential(*layers[:-1])
        params = (list(self.user_emb_.parameters()) + list(self.item_emb_.parameters())
                  + list(self.ub_.parameters()) + list(self.ib_.parameters())
                  + list(self.mlp_.parameters()))
        opt = nn.Adam(params, lr=self.lr)
        U = np.array([u for u, _, _ in triples])
        I = np.array([i for _, i, _ in triples])
        R = np.array([r for _, _, r in triples])
        self.global_mean_ = float(R.mean())
        n = len(R)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                pred = self._forward(U[b], I[b])
                ((pred - Tensor(R[b] - self.global_mean_)) ** 2).mean().backward()
                opt.step()
        return self

    def _forward(self, u, i):
        u = np.asarray(u); i = np.asarray(i)
        ue, ie = self.user_emb_(u), self.item_emb_(i)
        fm = (ue * ie).sum(axis=1) + self.ub_(u).reshape(-1) + self.ib_(i).reshape(-1)
        deep = self.mlp_(Tensor.concatenate([ue, ie], axis=1)).reshape(-1)
        return fm + deep                              # low-order FM + high-order deep

    def predict(self, user, item):
        if user >= self.n_users_ or item >= self.n_items_:
            return self.global_mean_
        return float(self._forward([user], [item]).data[0]) + self.global_mean_


__all__ = ["NeuralCF", "DeepFM"]
