"""Train neural members to DISAGREE, on purpose (Liu & Yao, 1999)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


def _mlp_params(dims, rng):
    from ..nn import Linear
    return [Linear(dims[i], dims[i + 1], rng=rng) for i in range(len(dims) - 1)]


def _mlp_forward(layers, x):
    h = x
    for i, l in enumerate(layers):
        h = l(h)
        if i < len(layers) - 1:
            h = h.relu()
    return h


class NegativeCorrelationLearning(BaseEstimator, RegressorMixin):
    """Train neural members to DISAGREE, on purpose (Liu & Yao, 1999).

    An ensemble only helps if its members make DIFFERENT errors, but training them
    independently gives no reason for that. Negative correlation learning trains all
    members TOGETHER with an extra penalty on each member's correlation with the
    ensemble's error -- so every net is pushed to specialise where the others are
    weak. The single knob ``lambda`` interpolates from independent training (0) to
    maximally diverse. It directly optimises the bias-variance-COVARIANCE
    decomposition that governs ensemble error. Small MLP members.
    """

    def __init__(self, n_estimators=5, hidden=16, lam=0.5, epochs=300, lr=0.02,
                 random_state=None):
        self.n_estimators = n_estimators
        self.hidden = hidden
        self.lam = lam
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float).reshape(-1, 1)
        rng = check_random_state(self.random_state)
        dims = [X.shape[1], self.hidden, 1]
        self.members_ = [_mlp_params(dims, rng) for _ in range(self.n_estimators)]
        params = [p for m in self.members_ for l in m for p in l.parameters()]
        xt, yt = Tensor(X), Tensor(y)
        for _ in range(self.epochs):
            outs = [_mlp_forward(m, xt) for m in self.members_]
            ens = outs[0]
            for o in outs[1:]:
                ens = ens + o
            ens = ens * (1.0 / self.n_estimators)
            loss = Tensor(np.zeros(1))
            for o in outs:
                mse = ((o - yt) ** 2).mean()
                penalty = ((o - ens) * (ens - yt)).mean()   # negative-correlation term
                loss = loss + mse + self.lam * penalty
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def predict(self, X):
        from ..autograd import Tensor
        X = check_array(X)
        outs = [_mlp_forward(m, Tensor(X)).data.ravel() for m in self.members_]
        return np.mean(outs, axis=0)


__all__ = ["NegativeCorrelationLearning"]
