"""Local regression: fit a different model near every point you are asked about.

THE IDEA
--------
There is no global model. To predict at ``x``, look at the training points near
``x``, fit a simple model to just those, evaluate it at ``x``, and throw it away.
Ask about a different ``x`` and the whole thing happens again.

That is a genuinely different way to think about fitting. A linear model has
parameters; LOESS has no parameters at all -- the training data IS the model. It
is "nonparametric" in the literal sense: nothing is summarised in advance, so the
answer is computed on demand.

THE PROGRESSION
---------------
Three methods, each one line of algebra from the last:

* ``NadarayaWatson``      -- weighted AVERAGE of nearby y. Fit a constant.
* ``LocalLinearRegression`` -- weighted LINE through nearby points. Fit a slope.
* ``LOESS``               -- local polynomial, plus optional robustness
  iterations against outliers.

Read them in that order; the second fixes a specific flaw in the first, and the
flaw is instructive.

BOUNDARY BIAS: WHY THE LINE BEATS THE AVERAGE
---------------------------------------------
Nadaraya-Watson averages the neighbours' ``y``. In the middle of the data, points
lie on both sides and their errors cancel. At the EDGE they do not: all the
neighbours are on one side, so if the function is sloping upward, every
neighbour of the leftmost point sits above it, and the average is biased upward.
The estimate flattens toward the boundary -- always, systematically, and worst
exactly where extrapolation makes people nervous anyway.

Fitting a local LINE removes it. The line has a slope, so it can follow the trend
through the boundary instead of averaging across it. This is not a small
refinement: local linear is boundary-bias-free to first order, which is why LOESS
is defined with a polynomial and Nadaraya-Watson is mostly a teaching device.

THE BANDWIDTH IS THE MODEL
--------------------------
``frac`` (or ``bandwidth``) decides how many neighbours count. It, not the
polynomial degree, controls the bias-variance trade-off:

* too small -- few points per fit, so the curve chases noise (high variance);
* too large -- the neighbourhood spans real structure, which gets averaged away
  (high bias); in the limit LOESS becomes ordinary linear regression.

There is nothing else to tune, and no amount of data makes the choice for you.
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


def _tricube(d):
    """The standard LOESS weight: ``(1 - |d|^3)^3`` for ``|d| < 1``, else 0.

    Two properties earn it the default. It reaches zero SMOOTHLY (value and
    first two derivatives vanish at the edge), so a point does not jerk into the
    fit as it enters the neighbourhood -- with a hard cutoff the fitted curve has
    visible kinks wherever the neighbour set changes. And it has COMPACT support,
    so only the neighbours are touched; a gaussian kernel technically weights
    every point in the dataset forever.
    """
    d = np.abs(d)
    return np.where(d < 1.0, (1.0 - d ** 3) ** 3, 0.0)


def _gaussian_kernel(d):
    return np.exp(-0.5 * d ** 2)


def _epanechnikov(d):
    """Minimises asymptotic mean squared error among kernels. The optimum is so
    shallow that the choice of kernel almost never matters -- which is itself
    worth knowing, since it means bandwidth is the only knob that does."""
    d = np.abs(d)
    return np.where(d < 1.0, 0.75 * (1.0 - d ** 2), 0.0)


_KERNELS = {"tricube": _tricube, "gaussian": _gaussian_kernel,
            "epanechnikov": _epanechnikov}


class NadarayaWatson(BaseEstimator, RegressorMixin):
    """Weighted average of nearby targets. The simplest possible smoother.

    ``prediction(x) = sum_i w_i(x) * y_i / sum_i w_i(x)``

    Read it as: every training point votes, weighted by how close it is. That is
    the whole method, and its flaw is structural -- see the module docstring on
    boundary bias. Use ``LocalLinearRegression`` instead; this is here because
    the comparison is the lesson.
    """

    def __init__(self, bandwidth=0.5, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        # "fitting" is only storage: a nonparametric model IS its training data
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        check_is_fitted(self, "X_")
        X = check_array(X)
        if self.kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel: {self.kernel!r}")
        kern = _KERNELS[self.kernel]

        # every query against every training point, in one broadcast
        dist = np.linalg.norm(X[:, None, :] - self.X_[None, :, :], axis=2)
        w = kern(dist / self.bandwidth)
        total = w.sum(axis=1)
        # a query with no neighbours at all: the kernel has compact support, so
        # this is reachable rather than theoretical. Fall back to the global mean
        # instead of dividing by zero
        out = np.full(len(X), self.y_.mean())
        ok = total > 1e-12
        out[ok] = (w[ok] @ self.y_) / total[ok]
        return out


class LocalLinearRegression(BaseEstimator, RegressorMixin):
    """A weighted least-squares LINE fitted afresh at every query point.

    Solves, for each query ``x0``::

        minimise  sum_i w_i(x0) * (y_i - b0 - b.(x_i - x0))^2

    and returns ``b0``, the fitted value at ``x0``. Centring the design on ``x0``
    is what makes the intercept the answer -- no evaluation step is needed, and
    the slope ``b`` falls out as a free local derivative estimate.

    Boundary-bias-free to first order, unlike ``NadarayaWatson``.
    """

    def __init__(self, bandwidth=0.5, kernel="tricube", ridge=1e-8):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.ridge = ridge

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        check_is_fitted(self, "X_")
        X = check_array(X)
        if self.kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel: {self.kernel!r}")
        kern = _KERNELS[self.kernel]
        out = np.empty(len(X))

        for i, x0 in enumerate(X):
            centred = self.X_ - x0
            w = kern(np.linalg.norm(centred, axis=1) / self.bandwidth)
            if w.sum() <= 1e-12:
                out[i] = self.y_.mean()
                continue
            # design centred on x0, so the intercept IS the prediction
            A = np.hstack([np.ones((len(self.X_), 1)), centred])
            # the ridge term is not regularisation for its own sake: with fewer
            # effective neighbours than features the normal equations are
            # singular, and a tiny diagonal keeps the solve from exploding
            # rather than changing the answer where it is well posed
            WA = A * w[:, None]
            beta = np.linalg.solve(A.T @ WA + self.ridge * np.eye(A.shape[1]),
                                   WA.T @ self.y_)
            out[i] = beta[0]
        return out


class LOESS(BaseEstimator, RegressorMixin):
    """Locally estimated scatterplot smoothing, with robustness iterations.

    LOCAL POLYNOMIAL
    ----------------
    ``degree=1`` is the local line above; ``degree=2`` fits a local parabola,
    which follows curvature (peaks and troughs) better at the cost of variance.
    Degree above 2 is essentially never worth it -- the bandwidth already
    controls flexibility, and a higher polynomial just adds noise.

    ``frac`` is the fraction of the data in each neighbourhood, so the
    neighbourhood is defined by RANK, not distance: it automatically widens where
    data is sparse and tightens where it is dense. That adaptivity is why LOESS
    is stated in terms of ``frac`` rather than a bandwidth in the units of x.

    THE ROBUSTNESS ITERATIONS
    -------------------------
    Least squares is not robust: one outlier drags the local fit toward itself,
    and because the fit is LOCAL, it drags the curve visibly at that x rather
    than being diluted across the whole dataset. A single bad point puts a bump
    in the curve.

    Cleveland's fix is to iterate. Fit, look at the residuals, and give each
    point an extra weight from the BISQUARE function of its residual -- points
    far off the fitted curve are downweighted toward zero, then refit. Two or
    three passes and a wild point has effectively removed itself.

    The scale is set by the MEDIAN absolute residual, not the mean, so the
    outlier cannot inflate the yardstick used to detect it. That detail is the
    whole reason the iteration converges to something sensible rather than
    slowly accepting the outlier.

    Cleveland (1979).
    """

    def __init__(self, frac=0.3, degree=1, n_iter=3, kernel="tricube",
                 ridge=1e-8):
        self.frac = frac
        self.degree = degree
        self.n_iter = n_iter
        self.kernel = kernel
        self.ridge = ridge

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        if X.shape[1] != 1:
            raise ValueError(
                "LOESS is implemented for a single feature; use "
                "LocalLinearRegression or GAM for more")
        self.X_, self.y_ = X, y

        # robustness weights are learned on the TRAINING data and then applied
        # at every query -- they are a property of each training point, not of
        # the point being predicted
        self.robust_weights_ = np.ones(len(y))
        for _ in range(max(0, self.n_iter - 1)):
            fitted = self._predict(X, self.robust_weights_)
            residual = y - fitted
            s = np.median(np.abs(residual))     # median: an outlier cannot
            if s <= 1e-12:                      # inflate its own yardstick
                break
            u = np.clip(residual / (6.0 * s), -1.0, 1.0)
            self.robust_weights_ = (1.0 - u ** 2) ** 2      # bisquare
        return self

    def _design(self, centred):
        """[1, d, d^2, ...] -- a polynomial in the centred coordinate."""
        return np.hstack([centred ** p for p in range(self.degree + 1)])

    def _predict(self, X, robust_w):
        kern = _KERNELS[self.kernel]
        n = len(self.X_)
        # frac defines the neighbourhood by RANK, so it adapts to local density
        k = max(self.degree + 2, int(np.ceil(self.frac * n)))
        out = np.empty(len(X))

        for i, x0 in enumerate(X):
            centred = self.X_ - x0
            dist = np.abs(centred[:, 0])
            # the k-th nearest distance sets the bandwidth for THIS query
            h = np.partition(dist, min(k - 1, n - 1))[min(k - 1, n - 1)]
            if h <= 0:
                h = np.max(dist) if np.max(dist) > 0 else 1.0
            w = kern(dist / h) * robust_w
            if w.sum() <= 1e-12:
                out[i] = self.y_.mean()
                continue
            A = self._design(centred)
            WA = A * w[:, None]
            try:
                beta = np.linalg.solve(A.T @ WA + self.ridge * np.eye(A.shape[1]),
                                       WA.T @ self.y_)
            except np.linalg.LinAlgError:
                out[i] = np.average(self.y_, weights=w)
                continue
            out[i] = beta[0]        # centred design: the intercept is the answer
        return out

    def predict(self, X):
        check_is_fitted(self, "X_")
        X = check_array(X)
        if self.kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel: {self.kernel!r}")
        return self._predict(X, self.robust_weights_)


__all__ = ["LOESS", "NadarayaWatson", "LocalLinearRegression"]
