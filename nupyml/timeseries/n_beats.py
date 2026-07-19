"""Deep basis-expansion forecasting, no recurrence (Oreshkin et al., 2020)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class NBeats(BaseEstimator):
    """Deep basis-expansion forecasting, no recurrence (Oreshkin et al., 2020).

    N-BEATS forecasts a window purely with fully-connected blocks. Each block reads
    the recent history, produces expansion coefficients, and expands them through a
    BASIS into a backcast (what it explains of the input) and a forecast; the
    backcast is subtracted so the next block models the residual. Stacking these --
    with trend (polynomial) and seasonal (Fourier) bases -- gives an interpretable,
    purely feed-forward forecaster that beat statistical baselines on the M4
    competition. Trained on sliding windows of one series.
    """

    def __init__(self, lookback=24, horizon=12, n_blocks=3, hidden=64, epochs=400,
                 lr=0.005, random_state=None):
        self.lookback = lookback
        self.horizon = horizon
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, y):
        from ..nn import Linear
        from ..autograd import Tensor
        y = np.asarray(y, float).ravel()
        self.y_ = y
        rng = check_random_state(self.random_state)
        L, H = self.lookback, self.horizon
        # build sliding windows
        Xs, Ys = [], []
        for i in range(len(y) - L - H + 1):
            Xs.append(y[i:i + L]); Ys.append(y[i + L:i + L + H])
        X = np.array(Xs); Y = np.array(Ys)
        self._mu, self._sd = X.mean(), X.std() + 1e-8
        Xn = (X - self._mu) / self._sd
        Yn = (Y - self._mu) / self._sd
        self.blocks_ = []
        for _ in range(self.n_blocks):
            self.blocks_.append([Linear(L, self.hidden, rng=rng),
                                 Linear(self.hidden, self.hidden, rng=rng),
                                 Linear(self.hidden, L, rng=rng),   # backcast
                                 Linear(self.hidden, H, rng=rng)])  # forecast
        params = [p for b in self.blocks_ for m in b for p in m.parameters()]
        xt, yt = Tensor(Xn), Tensor(Yn)
        for _ in range(self.epochs):
            forecast = self._run(xt)
            loss = ((forecast - yt) ** 2).mean()
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                if p.grad is not None:                   # last block's backcast is unused
                    p.data -= self.lr * p.grad
        return self

    def _run(self, x):
        residual = x
        forecast = None
        for b in self.blocks_:
            h = b[1](b[0](residual).relu()).relu()
            backcast = b[2](h)
            block_f = b[3](h)
            residual = residual - backcast               # doubly-residual stacking
            forecast = block_f if forecast is None else forecast + block_f
        return forecast

    def forecast(self, steps=None):
        from ..autograd import Tensor
        window = (self.y_[-self.lookback:] - self._mu) / self._sd
        out = self._run(Tensor(window.reshape(1, -1))).data.ravel()
        return out * self._sd + self._mu


__all__ = ["NBeats"]
