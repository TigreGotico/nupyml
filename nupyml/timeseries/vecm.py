"""Model series that share a long-run EQUILIBRIUM (Engle & Granger, 1987)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class VECM(BaseEstimator):
    """Model series that share a long-run EQUILIBRIUM (Engle & Granger, 1987).

    Two series can each wander like a random walk yet stay TIED TOGETHER -- a stock
    and its futures, two exchange rates -- so their SPREAD is stationary even though
    the levels are not. This is cointegration, and differencing them (as a VAR would)
    throws the relationship away. A vector error-correction model keeps it: it adds
    an ERROR-CORRECTION term ``alpha (beta'y_{t-1})`` that pulls the system back
    whenever the spread strays from equilibrium, on top of the short-run dynamics.
    Cointegration vector by Engle-Granger; the rest by OLS.
    """

    def __init__(self):
        pass

    def fit(self, Y):
        Y = check_array(Y)
        T, k = Y.shape
        # Engle-Granger: regress series 0 on the rest -> the cointegration vector
        X = np.column_stack([np.ones(T), Y[:, 1:]])
        gamma, *_ = np.linalg.lstsq(X, Y[:, 0], rcond=None)
        self.beta_ = np.concatenate([[1.0], -gamma[1:]])   # beta'Y = spread
        self.const_ = gamma[0]
        ect = Y @ self.beta_ - self.const_                 # error-correction term
        dY = np.diff(Y, axis=0)
        # dY_t = alpha * ect_{t-1} + Gamma * dY_{t-1} + eps
        rows, targ = [], []
        for t in range(1, len(dY)):
            rows.append(np.concatenate([[1.0], [ect[t]], dY[t - 1]]))
            targ.append(dY[t])
        A = np.array(rows); B = np.array(targ)
        self.coef_, *_ = np.linalg.lstsq(A, B, rcond=None)   # (2+k, k)
        self._Y = Y; self._ect = ect; self._dY = dY
        return self

    def forecast(self, steps=10):
        Y = list(self._Y)
        dY_last = self._dY[-1]
        out = []
        for _ in range(steps):
            ect = Y[-1] @ self.beta_ - self.const_
            feat = np.concatenate([[1.0], [ect], dY_last])
            dY_next = feat @ self.coef_
            Y.append(Y[-1] + dY_next)
            dY_last = dY_next
            out.append(Y[-1])
        return np.array(out)


__all__ = ["VECM"]
