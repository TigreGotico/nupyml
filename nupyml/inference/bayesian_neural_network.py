"""A dropout net read as an approximate BAYESIAN model (Gal & Ghahramani, 2016)."""
import numpy as np
from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array, check_random_state


class BayesianNeuralNetwork(BaseEstimator, RegressorMixin):
    """A dropout net read as an approximate BAYESIAN model (Gal & Ghahramani, 2016).

    Dropout is normally a training-time regulariser switched off at test time.
    Gal & Ghahramani showed that KEEPING dropout on at test time and averaging many
    stochastic forward passes is a valid approximation to Bayesian inference over
    the weights -- each pass is a sample from an approximate posterior. So a plain
    dropout MLP becomes a model that reports predictive UNCERTAINTY: the spread of
    its Monte-Carlo predictions widens where data is scarce (extrapolation) and
    narrows where it is dense. No extra parameters, just don't turn dropout off.
    """

    def __init__(self, hidden=(32, 32), dropout=0.1, epochs=400, lr=0.01,
                 n_samples=50, weight_decay=1e-4, random_state=None):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        from ..nn import Linear
        from ..autograd import Tensor
        X = check_array(X); y = np.asarray(y, float).reshape(-1, 1)
        rng = check_random_state(self.random_state)
        dims = [X.shape[1], *self.hidden, 1]
        self.layers_ = [Linear(dims[i], dims[i + 1], rng=rng)
                        for i in range(len(dims) - 1)]
        self._rng = rng
        params = [p for l in self.layers_ for p in l.parameters()]
        xt = Tensor(X); yt = Tensor(y)
        for _ in range(self.epochs):
            out = self._forward(xt, train=True)
            mse = ((out - yt) ** 2).mean()
            reg = sum((p * p).sum() for p in params) * self.weight_decay
            loss = mse + reg
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def _forward(self, x, train):
        from ..autograd import Tensor
        h = x
        for i, layer in enumerate(self.layers_):
            h = layer(h)
            if i < len(self.layers_) - 1:
                h = h.relu()
                if train or self.dropout > 0:            # dropout stays ON for MC
                    mask = (self._rng.rand(*h.shape) > self.dropout) / \
                        (1 - self.dropout)
                    h = h * Tensor(mask)
        return h

    def predict(self, X, return_std=False):
        from ..autograd import Tensor
        X = check_array(X)
        xt = Tensor(X)
        # Monte-Carlo: many stochastic passes with dropout left on
        preds = np.stack([self._forward(xt, train=True).data.ravel()
                          for _ in range(self.n_samples)])
        mean = preds.mean(axis=0)
        if not return_std:
            return mean
        return mean, preds.std(axis=0)


__all__ = ["BayesianNeuralNetwork"]
