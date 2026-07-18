"""Time series v4: seasonal ARIMA, trig-seasonal state space, hierarchy
reconciliation, and neural basis-expansion forecasting.

Four more forecasting tools. SARIMA adds seasonal differencing and seasonal AR/MA
to ARIMA. TBATS models complex/multiple seasonality with trigonometric terms and a
Box-Cox transform. Hierarchical reconciliation makes a set of forecasts across an
aggregation hierarchy COHERENT (the parts sum to the whole). N-BEATS forecasts with
a deep stack of basis-expansion blocks and no recurrence.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


def _difference(x, lag):
    return x[lag:] - x[:-lag]


class SARIMA(BaseEstimator):
    """Seasonal ARIMA: ARIMA that also models the SEASONAL rhythm (Box & Jenkins).

    ARIMA differences out a trend and regresses on recent lags, but it is blind to
    a yearly or weekly cycle. SARIMA adds a SEASONAL layer: it also differences at
    the season length ``m`` (removing the repeating pattern) and includes AR/MA
    terms at seasonal lags ``m, 2m, ...`` alongside the ordinary ones. So the model
    captures both "what happened last month" and "what happened this month last
    year". Orders ``(p, d, q)`` are regular, ``(P, D, Q, m)`` seasonal; fit by a
    Hannan-Rissanen-style least squares.
    """

    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.y_ = y
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        z = y.copy()
        for _ in range(d):
            z = _difference(z, 1)
        for _ in range(D):
            z = _difference(z, m)
        self._z_mean = z.mean()
        zc = z - self._z_mean
        # AR/MA lag sets: regular + seasonal
        ar_lags = list(range(1, p + 1)) + [m * k for k in range(1, P + 1)]
        ma_lags = list(range(1, q + 1)) + [m * k for k in range(1, Q + 1)]
        # stage 1: long AR to estimate residuals for the MA regressors
        resid = zc - self._ar_fit(zc, max(ar_lags + [1]) + 2)[1] if ma_lags else \
            np.zeros_like(zc)
        self.ar_lags_, self.ma_lags_ = ar_lags, ma_lags
        maxlag = max(ar_lags + ma_lags + [1])
        rows, targets = [], []
        for t in range(maxlag, len(zc)):
            feat = [zc[t - l] for l in ar_lags] + [resid[t - l] for l in ma_lags]
            rows.append([1.0] + feat); targets.append(zc[t])
        A = np.array(rows); b = np.array(targets)
        self.coef_, *_ = np.linalg.lstsq(A, b, rcond=None)
        self._resid = resid
        self._zc = zc
        return self

    @staticmethod
    def _ar_fit(x, order):
        rows, tgt = [], []
        for t in range(order, len(x)):
            rows.append(x[t - order:t][::-1]); tgt.append(x[t])
        A, b = np.array(rows), np.array(tgt)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = np.concatenate([np.zeros(order), A @ c])
        return c, pred

    def forecast(self, steps=10):
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        zc = list(self._zc); resid = list(self._resid)
        preds = []
        for _ in range(steps):
            feat = [1.0] + [zc[-l] for l in self.ar_lags_] \
                + [resid[-l] for l in self.ma_lags_]
            val = float(np.dot(self.coef_, feat))
            preds.append(val); zc.append(val); resid.append(0.0)
        z_fore = np.array(preds) + self._z_mean
        # invert seasonal then regular differencing
        hist = self.y_.copy()
        return self._integrate(hist, z_fore, d, D, m)

    def _integrate(self, y, dfore, d, D, m):
        # rebuild levels by cumulatively undoing the differences using tail history
        series = list(y)
        out = []
        for step, val in enumerate(dfore):
            base = 0.0
            if D:
                base += series[-m]
            if d:
                base += series[-1] - (series[-1 - m] if D else 0.0)
            level = val + base if (d or D) else val
            series.append(level); out.append(level)
        return np.array(out)


class TBATS(BaseEstimator):
    """Complex and MULTIPLE seasonality via trig terms + Box-Cox (De Livera, 2011).

    Holt-Winters needs one integer seasonal period. Real series can have several,
    possibly non-integer, cycles at once (daily + weekly, or a 365.25-day year).
    TBATS represents each season by a few TRIGONOMETRIC (Fourier) terms instead of a
    per-period state, so it handles high-frequency and multiple seasonalities
    cheaply, and a BOX-COX transform first stabilises a variance that grows with the
    level. Here: Box-Cox, then least-squares fit of a linear trend plus Fourier
    seasonal terms for each period, with the fit inverted for forecasting.
    """

    def __init__(self, periods=(12,), n_harmonics=3, box_cox_lambda=None):
        self.periods = list(periods)
        self.n_harmonics = n_harmonics
        self.box_cox_lambda = box_cox_lambda

    def _boxcox(self, y):
        lam = self.box_cox_lambda
        if lam is None or lam == 0:
            return np.log(y)
        return (y ** lam - 1) / lam

    def _inv_boxcox(self, z):
        lam = self.box_cox_lambda
        if lam is None or lam == 0:
            return np.exp(z)
        return (lam * z + 1) ** (1 / lam)

    def _design(self, t):
        cols = [np.ones_like(t), t]
        for p in self.periods:
            for k in range(1, self.n_harmonics + 1):
                cols.append(np.sin(2 * np.pi * k * t / p))
                cols.append(np.cos(2 * np.pi * k * t / p))
        return np.column_stack(cols)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        self.n_ = len(y)
        self._positive = np.all(y > 0)
        z = self._boxcox(y) if self._positive else y
        t = np.arange(self.n_, dtype=float)
        self.coef_, *_ = np.linalg.lstsq(self._design(t), z, rcond=None)
        return self

    def forecast(self, steps=10):
        t = np.arange(self.n_, self.n_ + steps, dtype=float)
        z = self._design(t) @ self.coef_
        return self._inv_boxcox(z) if self._positive else z


def reconcile_forecasts(base_forecasts, S, method="mint"):
    """Make hierarchical forecasts COHERENT -- the parts sum to the whole
    (Hyndman/Wickramasuriya).

    Forecast a total, its regions, and their stores separately and they will not
    add up. Reconciliation projects the independent ("base") forecasts onto the set
    that respects the hierarchy's summing constraints, encoded by the summing matrix
    ``S`` (each row = one node as a sum of the bottom-level series).

    * ``bottom_up`` -- trust only the leaves and sum them up.
    * ``ols`` -- least-squares projection onto the coherent subspace.
    * ``mint`` -- the minimum-trace optimal combination: the reconciliation with the
      smallest forecast-error variance (here with a diagonal error covariance).

    ``base_forecasts`` is ``(n_nodes,)`` or ``(n_nodes, horizon)``; returns coherent
    forecasts of the same shape.
    """
    S = np.asarray(S, float)
    yhat = np.atleast_2d(np.asarray(base_forecasts, float))
    if yhat.shape[0] != S.shape[0]:
        yhat = yhat.T
    n_bottom = S.shape[1]
    if method == "bottom_up":
        bottom = yhat[-n_bottom:]                        # leaves are the last rows
        P = np.hstack([np.zeros((n_bottom, S.shape[0] - n_bottom)), np.eye(n_bottom)])
    elif method == "ols":
        P = np.linalg.inv(S.T @ S) @ S.T
    elif method == "mint":
        W = np.diag(S.sum(axis=1))                       # diagonal error scale
        Wi = np.linalg.inv(W)
        P = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
    else:
        raise ValueError(f"unknown method {method!r}")
    reconciled = S @ (P @ yhat)
    return reconciled.ravel() if reconciled.shape[1] == 1 else reconciled


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


__all__ = ["SARIMA", "TBATS", "reconcile_forecasts", "NBeats"]
