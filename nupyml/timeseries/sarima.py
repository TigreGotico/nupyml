"""Seasonal ARIMA: ARIMA that also models the SEASONAL rhythm (Box & Jenkins)."""
import numpy as np
from ..base import BaseEstimator


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


__all__ = ["SARIMA"]
