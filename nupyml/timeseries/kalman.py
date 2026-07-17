"""The Kalman filter and its nonlinear descendants.

THE PROBLEM
-----------
A hidden state evolves; you see it only through noisy measurements. Track a
plane from radar blips, a robot from odometry, a price from trades. At each step
you have two sources of information that disagree:

* a PREDICTION of where the state should be, from the model of how it moves;
* a MEASUREMENT of where it seems to be, which is noisy.

Neither is right. The Kalman filter is the optimal way to combine them.

THE ONE IDEA
------------
Weight the two by their certainty. The KALMAN GAIN is exactly that weight::

    estimate = prediction + gain * (measurement - prediction)

* trust the measurement (small measurement noise) -> gain near 1 -> follow it;
* trust the model (small process noise) -> gain near 0 -> ignore the blip.

The filter tracks not just the estimate but its COVARIANCE -- how uncertain it
is -- and the gain falls out of the two covariances automatically. That is the
whole filter: predict (uncertainty grows), then update (a measurement shrinks
it), forever.

For a linear system with Gaussian noise this is not a heuristic -- it is provably
the optimal estimator, the one minimising mean squared error. That optimality is
why the same equations have flown every spacecraft since Apollo.

THE FAMILY
----------
Reality is rarely linear, so:

* ``ExtendedKalmanFilter`` -- LINEARISE the dynamics at the current estimate (take
  the Jacobian) and run the ordinary filter on that. Works when the nonlinearity
  is mild; fails, sometimes silently, when it is sharp, because a tangent line is
  a poor model of a sharp curve.
* ``UnscentedKalmanFilter`` -- do not linearise. Push a few carefully chosen
  "sigma points" through the true nonlinear function and read the mean and
  covariance off where they land. Captures the nonlinearity to higher order, with
  no derivatives to derive.
* ``ParticleFilter`` -- abandon the Gaussian entirely. Represent the state
  distribution by a cloud of weighted samples, and let it be any shape at all --
  multimodal, skewed, bounded. The most general and the most expensive.

Read them in that order: each drops an assumption the last relied on -- linear,
then linearisable, then Gaussian -- and pays for it in cost.

Kalman (1960); Rauch, Tung & Striebel (1965).
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class KalmanFilter(BaseEstimator):
    """The linear-Gaussian Kalman filter and RTS smoother.

    The model::

        x_t = F x_{t-1} + process noise   (Q)     how the state moves
        z_t = H x_t     + measurement noise (R)   how it is observed

    ``F`` is the state transition, ``H`` the observation map, ``Q`` and ``R`` the
    two noise covariances. Everything the filter does is driven by the tension
    between ``Q`` (how much the state can drift) and ``R`` (how noisy the sensor
    is).
    """

    def __init__(self, F, H, Q, R, x0=None, P0=None):
        self.F = np.atleast_2d(F)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x0 = x0
        self.P0 = P0

    def _init_state(self):
        n = self.F.shape[0]
        x = np.zeros(n) if self.x0 is None else np.asarray(self.x0, float)
        P = np.eye(n) if self.P0 is None else np.atleast_2d(self.P0)
        return x, P.copy()

    def filter(self, Z):
        """Forward pass: estimate each state from observations UP TO that time.

        Causal -- usable in real time. Returns filtered means and covariances,
        and stashes the one-step predictions the smoother will need.
        """
        Z = np.asarray(Z, float)
        n_obs = self.H.shape[0]
        # accept a 1D series of scalar observations as a column of measurements
        Z = Z.reshape(-1, n_obs)
        x, P = self._init_state()
        n_steps = len(Z)

        means = np.zeros((n_steps, len(x)))
        covs = np.zeros((n_steps, len(x), len(x)))
        # the predictions are kept because the RTS smoother walks backward
        # through exactly them -- recomputing would be wasteful and error-prone
        self._pred_means = np.zeros((n_steps, len(x)))
        self._pred_covs = np.zeros((n_steps, len(x), len(x)))

        for t in range(n_steps):
            # PREDICT: move the state forward, and let uncertainty grow by Q
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            self._pred_means[t], self._pred_covs[t] = x, P

            # UPDATE: fold in the measurement, weighted by the Kalman gain
            y = Z[t] - self.H @ x               # innovation: what surprised us
            S = self.H @ P @ self.H.T + self.R  # its covariance
            K = P @ self.H.T @ np.linalg.inv(S)  # the gain: prediction vs measure
            x = x + K @ y
            # Joseph form would be more stable; this plain form is clearer and
            # fine for well-conditioned R
            P = (np.eye(len(x)) - K @ self.H) @ P
            means[t], covs[t] = x, P

        self.filtered_means_ = means
        self.filtered_covs_ = covs
        return means, covs

    def smooth(self, Z):
        """RTS smoother: estimate each state from the ENTIRE series.

        Runs the forward filter, then a backward pass that corrects each estimate
        using everything that came after it. Strictly more accurate than
        filtering -- it uses future data -- and therefore only usable offline.

        The backward gain plays the mirror role of the Kalman gain: it says how
        much to trust the smoothed future over the filtered present.
        """
        means, covs = self.filter(Z)
        n_steps = len(means)
        sm_means = means.copy()
        sm_covs = covs.copy()

        for t in range(n_steps - 2, -1, -1):
            # how the filtered state at t relates to the prediction of t+1
            C = covs[t] @ self.F.T @ np.linalg.inv(self._pred_covs[t + 1])
            sm_means[t] = means[t] + C @ (sm_means[t + 1] - self._pred_means[t + 1])
            sm_covs[t] = covs[t] + C @ (sm_covs[t + 1] - self._pred_covs[t + 1]) @ C.T

        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs
        return sm_means, sm_covs

    def predict(self, Z, n_ahead=1):
        """Forecast ``n_ahead`` steps past the end of the series.

        With no measurements to correct it, forecasting is pure prediction: the
        state coasts on ``F`` and the uncertainty grows by ``Q`` every step,
        without bound. That widening is honest -- the further out, the less the
        model knows -- and is the state-space answer to "how far can I trust this
        forecast".
        """
        means, covs = self.filter(Z)
        x, P = means[-1], covs[-1]
        out = np.zeros((n_ahead, len(x)))
        out_cov = np.zeros((n_ahead, len(x), len(x)))
        for t in range(n_ahead):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            out[t], out_cov[t] = x, P
        return out, out_cov


class ExtendedKalmanFilter(BaseEstimator):
    """Kalman for nonlinear dynamics, by linearising at each step.

    ``f(x)`` and ``h(x)`` are the nonlinear transition and observation; ``F_jac``
    and ``H_jac`` return their Jacobians. At each step the filter evaluates the
    true nonlinear function to propagate the MEAN, but uses the Jacobian -- the
    local tangent -- to propagate the COVARIANCE.

    That split is the EKF's whole nature, and its weakness. A tangent line is only
    a good model of a curve nearby; if the state is uncertain enough that the
    nonlinearity bends appreciably across its spread, the linearised covariance is
    wrong, and the filter can diverge while reporting shrinking uncertainty --
    confidently lost. When that happens the unscented filter is the fix.
    """

    def __init__(self, f, h, F_jac, H_jac, Q, R, x0, P0=None):
        self.f, self.h = f, h
        self.F_jac, self.H_jac = F_jac, H_jac
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x0 = np.asarray(x0, float)
        self.P0 = P0

    def filter(self, Z):
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        x = self.x0.copy()
        P = np.eye(len(x)) if self.P0 is None else np.atleast_2d(self.P0).copy()
        means = np.zeros((len(Z), len(x)))
        covs = np.zeros((len(Z), len(x), len(x)))

        for t in range(len(Z)):
            # mean by the true f; covariance by its Jacobian -- the EKF split
            F = np.atleast_2d(self.F_jac(x))
            x = np.atleast_1d(self.f(x))
            P = F @ P @ F.T + self.Q

            H = np.atleast_2d(self.H_jac(x))
            y = Z[t] - np.atleast_1d(self.h(x))
            S = H @ P @ H.T + self.R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(len(x)) - K @ H) @ P
            means[t], covs[t] = x, P

        self.filtered_means_ = means
        self.filtered_covs_ = covs
        return means, covs


class UnscentedKalmanFilter(BaseEstimator):
    """Kalman for nonlinear dynamics WITHOUT linearising.

    THE UNSCENTED TRANSFORM
    -----------------------
    The insight: it is easier to approximate a distribution than an arbitrary
    function. Rather than linearising ``f``, choose ``2n+1`` SIGMA POINTS that
    capture the state's mean and covariance exactly, push each one through the
    true nonlinear ``f``, and read the new mean and covariance off where they
    land.

    Because the sigma points go through the real function, curvature is captured
    to second order -- markedly better than the EKF's tangent line -- and no
    Jacobian is ever needed, which matters when ``f`` is a black box or an
    awkward derivative. "Unscented" is just Julier's coinage; the mechanism is
    deterministic sampling, a close cousin of the particle filter with a handful
    of cleverly placed points instead of a random cloud.

    Julier & Uhlmann (1997).
    """

    def __init__(self, f, h, Q, R, x0, P0=None, alpha=1e-3, beta=2.0, kappa=0.0):
        self.f, self.h = f, h
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x0 = np.asarray(x0, float)
        self.P0 = P0
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

    def _sigma_points(self, x, P):
        n = len(x)
        lam = self.alpha ** 2 * (n + self.kappa) - n
        # matrix square root via Cholesky: the columns are the directions to
        # spread the sigma points along, scaled by the covariance
        S = np.linalg.cholesky((n + lam) * P)
        pts = [x]
        for i in range(n):
            pts.append(x + S[:, i])
            pts.append(x - S[:, i])
        # two weight sets: the mean weight and the covariance weight differ only
        # for the centre point (the beta term folds in knowledge that the true
        # distribution is roughly Gaussian)
        wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        return np.array(pts), wm, wc

    def _unscented_transform(self, sigmas, wm, wc, noise):
        mean = wm @ sigmas
        d = sigmas - mean
        cov = (wc[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) + noise
        return mean, cov

    def filter(self, Z):
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        x = self.x0.copy()
        P = np.eye(len(x)) if self.P0 is None else np.atleast_2d(self.P0).copy()
        means = np.zeros((len(Z), len(x)))
        covs = np.zeros((len(Z), len(x), len(x)))

        for t in range(len(Z)):
            # PREDICT: push sigma points through f, recover mean and covariance
            sigmas, wm, wc = self._sigma_points(x, P)
            prop = np.array([np.atleast_1d(self.f(s)) for s in sigmas])
            x, P = self._unscented_transform(prop, wm, wc, self.Q)

            # UPDATE: push the (re-drawn) sigma points through h
            sigmas, wm, wc = self._sigma_points(x, P)
            meas = np.array([np.atleast_1d(self.h(s)) for s in sigmas])
            z_mean, S = self._unscented_transform(meas, wm, wc, self.R)
            # cross-covariance between state and measurement gives the gain
            dx = sigmas - x
            dz = meas - z_mean
            Pxz = (wc[:, None, None] * dx[:, :, None] * dz[:, None, :]).sum(0)
            K = Pxz @ np.linalg.inv(S)
            x = x + K @ (Z[t] - z_mean)
            P = P - K @ S @ K.T
            means[t], covs[t] = x, P

        self.filtered_means_ = means
        self.filtered_covs_ = covs
        return means, covs


class ParticleFilter(BaseEstimator):
    """A filter for when nothing is Gaussian: represent the state by samples.

    THE IDEA
    --------
    Drop every distributional assumption. Represent the state's distribution by a
    cloud of ``n_particles`` weighted samples, and let it be any shape at all --
    two-humped, skewed, bounded, whatever the problem produces. Each step:

    1. **Propagate.** Push every particle through the (possibly nonlinear,
       possibly non-Gaussian) dynamics, with noise.
    2. **Weight.** Weight each particle by how well it explains the new
       measurement -- its likelihood.
    3. **Resample.** Draw a new set of particles in proportion to the weights.

    Step 3 is the crux. Without it, a few particles accumulate almost all the
    weight while the rest drift into irrelevance -- "degeneracy", where you have
    a thousand particles but two that matter. Resampling concentrates the cloud
    where the probability is, spending particles where they count.

    THE COST AND THE CURSE
    ----------------------
    The most general filter here, and the most expensive: accuracy improves only
    as ``1/sqrt(n_particles)``, so halving the error costs four times the
    particles. And the number needed grows exponentially with state dimension --
    the same curse that afflicts every sampling method. Excellent in low
    dimensions with nasty nonlinearity; hopeless in high dimensions.

    Gordon, Salmond & Smith (1993).
    """

    def __init__(self, transition, likelihood, n_particles=1000,
                 init_sampler=None, random_state=None):
        self.transition = transition        # (particles, rng) -> new particles
        self.likelihood = likelihood        # (particles, z) -> weights
        self.n_particles = n_particles
        self.init_sampler = init_sampler     # (n, rng) -> initial particles
        self.random_state = random_state

    def _resample(self, particles, weights, rng):
        """Systematic resampling: lower variance than plain multinomial.

        One uniform draw seeds a comb of evenly-spaced pointers into the
        cumulative weights. It cannot, by construction, drop a heavy particle the
        way independent draws occasionally do, so it adds less noise per step --
        which matters because resampling noise accumulates over the whole series.
        """
        n = len(particles)
        positions = (rng.uniform() + np.arange(n)) / n
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0            # guard against rounding leaving a gap at 1
        idx = np.searchsorted(cumsum, positions)
        return particles[idx]

    def filter(self, Z):
        Z = np.asarray(Z, float)
        rng = check_random_state(self.random_state)
        if self.init_sampler is not None:
            particles = self.init_sampler(self.n_particles, rng)
        else:
            particles = rng.normal(size=(self.n_particles, 1))

        means = []
        self.ess_ = []              # effective sample size, the degeneracy gauge
        for t in range(len(Z)):
            particles = self.transition(particles, rng)
            w = self.likelihood(particles, Z[t])
            w = w / (w.sum() + 1e-300)
            # the estimate is the weighted mean of the cloud
            means.append((w[:, None] * particles).sum(0))

            # effective sample size: near n_particles when weights are even, near
            # 1 when one particle dominates. Resampling when it drops is the
            # standard guard against wasting the cloud
            ess = 1.0 / (w ** 2).sum()
            self.ess_.append(ess)
            if ess < self.n_particles / 2:
                particles = self._resample(particles, w, rng)

        self.filtered_means_ = np.array(means)
        return self.filtered_means_


__all__ = ["KalmanFilter", "ExtendedKalmanFilter", "UnscentedKalmanFilter",
           "ParticleFilter"]
