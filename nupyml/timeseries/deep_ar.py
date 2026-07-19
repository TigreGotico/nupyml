"""Forecast a whole predictive DISTRIBUTION, not a mean (Salinas et al., 2020)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class DeepAR(BaseEstimator):
    """Forecast a whole predictive DISTRIBUTION, not a mean (Salinas et al., 2020).

    Most forecasters output a point; a decision-maker needs the RANGE. DeepAR trains
    an autoregressive network to output, at each step, the parameters of a
    distribution (here a Gaussian mean and variance) conditioned on the recent
    history, fit by maximising the Gaussian likelihood. Sampling it forward gives
    probabilistic forecasts -- prediction intervals with calibrated coverage -- which
    is what inventory, capacity and risk decisions actually require. A small MLP over
    a lag window here.
    """

    def __init__(self, lookback=10, hidden=32, epochs=300, lr=0.01,
                 random_state=None):
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, y):
        from ..nn import Linear
        from ..autograd import Tensor
        y = np.asarray(y, float).ravel()
        self.y_ = y
        self._mu, self._sd = y.mean(), y.std() + 1e-8
        yn = (y - self._mu) / self._sd
        rng = check_random_state(self.random_state)
        L = self.lookback
        X = np.array([yn[i:i + L] for i in range(len(yn) - L)])
        target = yn[L:]
        self.l1 = Linear(L, self.hidden, rng=rng)
        self.l2m = Linear(self.hidden, 1, rng=rng)
        self.l2s = Linear(self.hidden, 1, rng=rng)
        params = (list(self.l1.parameters()) + list(self.l2m.parameters())
                  + list(self.l2s.parameters()))
        xt, tt = Tensor(X), Tensor(target.reshape(-1, 1))
        for _ in range(self.epochs):
            h = self.l1(xt).relu()
            mean = self.l2m(h)
            log_sig = self.l2s(h)
            sig2 = (2.0 * log_sig).exp() + 1e-6
            nll = (0.5 * ((tt - mean) ** 2 / sig2 + (2.0 * log_sig))).mean()
            for p in params:
                p.zero_grad()
            nll.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def _step(self, window):
        from ..autograd import Tensor
        h = self.l1(Tensor(window.reshape(1, -1))).relu()
        mean = self.l2m(h).data.ravel()[0]
        sig = np.exp(self.l2s(h).data.ravel()[0])
        return mean, sig

    def forecast(self, steps=10, n_samples=100, random_state=None):
        rng = check_random_state(random_state)
        paths = []
        for _ in range(n_samples):
            window = ((self.y_[-self.lookback:] - self._mu) / self._sd).copy()
            path = []
            for _ in range(steps):
                m, s = self._step(window)
                val = m + s * rng.randn()
                path.append(val)
                window = np.append(window[1:], val)
            paths.append(np.array(path) * self._sd + self._mu)
        paths = np.array(paths)
        return paths.mean(axis=0), paths


__all__ = ["DeepAR"]
