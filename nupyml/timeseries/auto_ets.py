"""Pick the exponential-smoothing model by information criterion (Hyndman, 2002)."""
import numpy as np
from ..base import BaseEstimator, RegressorMixin


class AutoETS(BaseEstimator, RegressorMixin):
    """Pick the exponential-smoothing model by information criterion (Hyndman, 2002).

    Exponential smoothing comes in a family -- with or without a trend, with or
    without a season, each additive or damped. AutoETS fits several of these
    ERROR-TREND-SEASONAL forms and selects the one with the best AIC, so you get
    the appropriate model instead of guessing. It is the engine behind R's
    ``ets()`` / ``forecast`` defaults. Additive components; ``season_length=1``
    disables seasonality.
    """

    def __init__(self, season_length=1):
        self.season_length = season_length

    def _fit_ets(self, y, trend, season):
        m = self.season_length
        n = len(y)
        # grid over smoothing params; additive components
        best = None
        alphas = np.linspace(0.1, 0.9, 5)
        betas = np.linspace(0.05, 0.5, 4) if trend else [0.0]
        gammas = np.linspace(0.05, 0.5, 4) if season else [0.0]
        for a in alphas:
            for b in betas:
                for g in gammas:
                    sse, params = self._run(y, a, b, g, trend, season)
                    k = 1 + trend + season + 1
                    aic = n * np.log(sse / n + 1e-12) + 2 * k
                    if best is None or aic < best[0]:
                        best = (aic, a, b, g, trend, season, params)
        return best

    def _run(self, y, a, b, g, trend, season):
        m = self.season_length
        level = y[0]
        tr = (y[m] - y[0]) / m if trend and len(y) > m else 0.0
        seas = list(y[:m] - y[:m].mean()) if season else [0.0] * max(m, 1)
        sse = 0.0
        for t in range(len(y)):
            s = seas[t % len(seas)] if season else 0.0
            fitted = level + (tr if trend else 0.0) + s
            e = y[t] - fitted
            sse += e * e
            new_level = a * (y[t] - s) + (1 - a) * (level + (tr if trend else 0.0))
            if trend:
                tr = b * (new_level - level) + (1 - b) * tr
            if season:
                seas[t % len(seas)] = g * (y[t] - new_level) + (1 - g) * s
            level = new_level
        return sse, (level, tr, seas)

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        cands = [self._fit_ets(y, False, False)]
        if len(y) > 3:
            cands.append(self._fit_ets(y, True, False))
        if self.season_length > 1 and len(y) > 2 * self.season_length:
            cands.append(self._fit_ets(y, True, True))
            cands.append(self._fit_ets(y, False, True))
        self.best_ = min(cands, key=lambda c: c[0])
        _, _, _, _, self.trend_, self.season_, self.state_ = self.best_
        return self

    def forecast(self, steps=10):
        level, tr, seas = self.state_
        out = np.empty(steps)
        for h in range(steps):
            s = seas[h % len(seas)] if self.season_ else 0.0
            out[h] = level + ((h + 1) * tr if self.trend_ else 0.0) + s
        return out


__all__ = ["AutoETS"]
