"""Time series v5: cointegration, latent factors, intermittent-demand variants,
state-space EM, and probabilistic autoregression.

Five more forecasting tools. VECM models series that share a long-run equilibrium.
The dynamic factor model compresses many series into a few common drivers. The
Croston SBA/TSB variants fix its known biases for intermittent demand. Kalman-EM
learns a state-space model's noise levels from data. DeepAR forecasts a whole
predictive DISTRIBUTION, not just a mean.
"""
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


class DynamicFactorModel(BaseEstimator):
    """Compress many series into a few COMMON factors (Geweke, 1977).

    Dozens of macro indicators, or hundreds of sensors, are mostly driven by a
    handful of shared forces -- a business cycle, a temperature field. A dynamic
    factor model extracts a few latent FACTORS (here via principal components of the
    standardised series) that explain their comovement, models the factors' own
    dynamics with a VAR, and reconstructs every series as a loading on those factors
    plus idiosyncratic noise. Forecasting the few factors and mapping back forecasts
    all series at once, with far fewer parameters than a full VAR. ``n_factors`` is
    the number of common drivers.
    """

    def __init__(self, n_factors=2, var_order=1):
        self.n_factors = n_factors
        self.var_order = var_order

    def fit(self, Y):
        Y = check_array(Y)
        self.mean_ = Y.mean(axis=0); self.std_ = Y.std(axis=0) + 1e-8
        Z = (Y - self.mean_) / self.std_
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self.loadings_ = Vt[:self.n_factors]               # (r, k)
        F = Z @ self.loadings_.T                            # factors (T, r)
        self.factors_ = F
        # VAR(1) on the factors
        rows = np.column_stack([np.ones(len(F) - 1), F[:-1]])
        self.var_coef_, *_ = np.linalg.lstsq(rows, F[1:], rcond=None)
        return self

    def forecast(self, steps=10):
        f = self.factors_[-1].copy()
        out = []
        for _ in range(steps):
            f = np.concatenate([[1.0], f]) @ self.var_coef_
            z = f @ self.loadings_                          # reconstruct series
            out.append(z * self.std_ + self.mean_)
        return np.array(out)


class CrostonSBA(BaseEstimator):
    """Croston with the Syntetos-Boylan bias correction (Syntetos & Boylan, 2005).

    Croston's method forecasts intermittent demand as ``size / interval``, but that
    ratio is BIASED UPWARD (the expectation of a ratio is not the ratio of
    expectations). SBA multiplies the forecast by ``1 - alpha/2`` -- a small,
    principled correction that removes most of the bias and, on the standard
    intermittent-demand benchmarks, beats plain Croston. ``alpha`` smooths both the
    demand sizes and the gaps between them.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        nz = np.where(y > 0)[0]
        if len(nz) == 0:
            self.rate_ = 0.0
            return self
        size = y[nz[0]]; interval = nz[0] + 1
        last = nz[0]
        for i in nz[1:]:
            gap = i - last
            size += self.alpha * (y[i] - size)
            interval += self.alpha * (gap - interval)
            last = i
        self.size_, self.interval_ = size, max(interval, 1e-6)
        self.rate_ = (1 - self.alpha / 2) * size / self.interval_   # SBA correction
        return self

    def predict(self, steps=1):
        return np.full(steps, self.rate_)


class CrostonTSB(BaseEstimator):
    """Croston that tracks a demand PROBABILITY, not an interval (Teunter, 2011).

    Croston never updates its forecast during a run of zeros, so it cannot react to
    demand that is DYING OUT (obsolescence). TSB replaces the inter-demand interval
    with a demand PROBABILITY that is updated EVERY period -- decayed on a zero,
    bumped on a sale -- and forecasts ``probability * size``. Because it updates on
    zeros too, it correctly lets the forecast fade for a discontinued item, which is
    exactly where Croston fails. ``alpha`` smooths sizes, ``beta`` the probability.
    """

    def __init__(self, alpha=0.1, beta=0.05):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        nz = y > 0
        p = nz.mean()
        size = y[nz].mean() if nz.any() else 0.0
        for t in range(len(y)):
            if y[t] > 0:
                p += self.beta * (1 - p)
                size += self.alpha * (y[t] - size)
            else:
                p += self.beta * (0 - p)                    # decays during zeros
        self.prob_, self.size_ = p, size
        self.rate_ = p * size
        return self

    def predict(self, steps=1):
        return np.full(steps, self.rate_)


class KalmanEM(BaseEstimator):
    """Learn a state-space model's NOISE levels from data (Shumway & Stoffer, 1982).

    A Kalman filter needs to know the process and observation noise variances, but
    you rarely do. Kalman-EM estimates them: the E-step runs the filter and RTS
    smoother to get the expected states, the M-step re-estimates the variances from
    those expectations, and iterating converges to the maximum-likelihood noise
    levels -- so the model calibrates its OWN uncertainty. Fitted here for a local-
    level (random-walk-plus-noise) model; ``signal_to_noise_`` is the learned ratio.
    """

    def __init__(self, max_iter=50, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        n = len(y)
        q = np.var(np.diff(y)) / 2 + 1e-6                   # process var (init)
        r = np.var(y) / 2 + 1e-6                            # obs var (init)
        prev = -np.inf
        for _ in range(self.max_iter):
            # forward filter
            a = np.zeros(n); P = np.zeros(n)
            a[0] = y[0]; P[0] = r
            for t in range(1, n):
                ap = a[t - 1]; Pp = P[t - 1] + q
                K = Pp / (Pp + r)
                a[t] = ap + K * (y[t] - ap); P[t] = (1 - K) * Pp
            # RTS smoother (store the gains J for the lag-one covariance)
            asm = a.copy(); Psm = P.copy(); Js = np.zeros(n)
            for t in range(n - 2, -1, -1):
                Pp = P[t] + q
                J = P[t] / Pp; Js[t] = J
                asm[t] = a[t] + J * (asm[t + 1] - a[t])
                Psm[t] = P[t] + J * J * (Psm[t + 1] - Pp)
            # M-step: full EM variances including the smoothed-state uncertainty
            # E[(mu_t - mu_{t-1})^2] = (dmu)^2 + Psm[t] + Psm[t-1] - 2 cov(mu_t,mu_{t-1})
            cross = Js[:-1] * Psm[1:]
            q = np.mean(np.diff(asm) ** 2 + Psm[1:] + Psm[:-1] - 2 * cross) + 1e-9
            r = np.mean((y - asm) ** 2 + Psm)
            ll = -0.5 * np.sum((y - asm) ** 2) / r
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.process_var_, self.obs_var_ = q, r
        self.signal_to_noise_ = q / r
        self.level_ = asm
        return self

    def forecast(self, steps=10):
        return np.full(steps, self.level_[-1])


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


__all__ = ["VECM", "DynamicFactorModel", "CrostonSBA", "CrostonTSB", "KalmanEM",
           "DeepAR"]
