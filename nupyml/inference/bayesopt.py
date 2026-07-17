"""Bayesian optimization: minimise a function that is expensive to evaluate.

THE SETTING
-----------
Some objectives cost a fortune per evaluation: training a big model to score one
hyperparameter setting, running a physical experiment, a week of simulation. You
get a few dozen evaluations, not millions, so grid search and random search --
which spend their budget blindly -- are wasteful. Every query must count.

THE STRATEGY
------------
Two pieces working together:

1. **A surrogate model** -- a Gaussian process -- that fits everything seen so
   far and, crucially, reports its UNCERTAINTY everywhere it has not looked.
2. **An acquisition function** that reads the surrogate and proposes the single
   most VALUABLE next point to evaluate.

Evaluate there, update the surrogate, repeat. Cheap reasoning about the surrogate
replaces expensive evaluations of the real objective -- you think hard to query
rarely.

WHY A GAUSSIAN PROCESS
----------------------
The surrogate must say not just "I predict 5 here" but "I predict 5, and I am
very unsure". A GP gives exactly that: a mean AND a calibrated variance at every
point, with the variance GROWING away from observed data. That honest uncertainty
is what the acquisition function steers by, and it is why the GP -- not a random
forest or a neural net -- is the canonical surrogate.

THE ACQUISITION TRADE-OFF
-------------------------
Every acquisition function balances the same two impulses:

* EXPLOIT -- query where the surrogate predicts a good value;
* EXPLORE -- query where the surrogate is uncertain, in case something better
  hides there.

* ``EI`` (Expected Improvement) -- the expected amount by which a point beats the
  best so far. Balances the two automatically, and is the standard default.
* ``UCB`` (Upper/Lower Confidence Bound) -- ``mean - kappa * std``; ``kappa``
  tunes the balance by hand.
* ``PI`` (Probability of Improvement) -- probability of beating the best. Tends to
  under-explore, chasing tiny sure gains over large uncertain ones.

Mockus (1978); Snoek, Larochelle & Adams (2012).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import norm

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """A GP with an RBF kernel: predictions WITH calibrated uncertainty.

    THE ONE THING IT DOES THAT OTHER REGRESSORS DO NOT
    --------------------------------------------------
    Return a variance that GROWS away from the data. Near an observed point the GP
    is confident; far from all of them it reverts to the prior and says so, with a
    wide band. That property -- knowing what it does not know -- is why the GP is
    the surrogate for Bayesian optimization and the tool of choice whenever the
    uncertainty itself is the product. (Contrast the RVM, whose variance famously
    SHRINKS far from the data -- see ``nonparametric``.)

    The mechanics are exact Bayesian linear regression in the kernel's feature
    space: a Cholesky solve gives the posterior mean and covariance in closed
    form. ``length_scale`` sets how far correlations reach -- how quickly the
    function is assumed to wiggle -- and is the parameter that matters most.
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise = noise

    def _kernel(self, A, B):
        # RBF: correlation decays with squared distance over the length scale
        sq = cdist(A, B, "sqeuclidean")
        return self.signal_variance * np.exp(-0.5 * sq / self.length_scale ** 2)

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_train_ = X
        self.y_mean_ = y.mean()
        yc = y - self.y_mean_
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        # Cholesky once; both the mean and the variance reuse the factor
        self._L = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._L, yc)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "X_train_")
        X = check_array(X)
        K_star = self._kernel(X, self.X_train_)
        mean = K_star @ self._alpha + self.y_mean_
        if not return_std:
            return mean
        # posterior variance = prior variance minus what the data explains; the
        # subtracted term shrinks far from the data, so the variance grows there
        v = cho_solve(self._L, K_star.T)
        var = self.signal_variance - np.sum(K_star * v.T, axis=1)
        return mean, np.sqrt(np.maximum(var, 1e-12))


def expected_improvement(mean, std, best, xi=0.01):
    """Expected amount by which a point improves on ``best`` (for minimisation).

    The default acquisition because it balances explore and exploit with no knob:
    it is large where the mean is good (exploit) OR where the std is high and
    could reach below best (explore), and the expectation weighs the two
    correctly. ``xi`` adds a small margin that nudges toward more exploration.
    """
    std = np.maximum(std, 1e-9)
    improvement = best - mean - xi          # positive where we expect to improve
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def upper_confidence_bound(mean, std, kappa=2.0):
    """``mean - kappa * std`` (minimisation): optimistic about low values.

    ``kappa`` is the explore/exploit dial made explicit -- 0 is pure greed, large
    is aggressive exploration. Unlike EI it must be tuned, but it is trivial to
    reason about, which is why it is the pedagogical favourite.
    """
    return mean - kappa * std


def probability_of_improvement(mean, std, best, xi=0.01):
    """Probability of beating ``best``. Under-explores -- see the module docstring."""
    std = np.maximum(std, 1e-9)
    return norm.cdf((best - mean - xi) / std)


class BayesianOptimization(BaseEstimator):
    """Minimise an expensive black box in few evaluations.

    ``bounds`` is ``(lo, hi)`` per dimension. ``maximize`` searches for a maximum
    instead (negated internally). After ``run``, ``best_x_`` and ``best_y_`` hold
    the best point found and ``history_`` the whole trace.
    """

    def __init__(self, func, bounds, acquisition="ei", n_init=5, n_iter=25,
                 length_scale=1.0, kappa=2.0, xi=0.01, maximize=False,
                 n_candidates=1000, random_state=None):
        self.func = func
        self.bounds = np.asarray(bounds, float)
        self.acquisition = acquisition
        self.n_init = n_init
        self.n_iter = n_iter
        self.length_scale = length_scale
        self.kappa = kappa
        self.xi = xi
        self.maximize = maximize
        self.n_candidates = n_candidates
        self.random_state = random_state

    def _acquire(self, mean, std, best):
        if self.acquisition == "ei":
            # higher is better for EI/PI, so negate to keep "argmin" throughout
            return -expected_improvement(mean, std, best, self.xi)
        if self.acquisition == "ucb":
            return upper_confidence_bound(mean, std, self.kappa)
        if self.acquisition == "pi":
            return -probability_of_improvement(mean, std, best, self.xi)
        raise ValueError(f"Unknown acquisition: {self.acquisition!r}")

    def run(self):
        rng = check_random_state(self.random_state)
        sign = -1.0 if self.maximize else 1.0
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        dim = len(self.bounds)

        # seed with random points: the GP needs a few observations before its
        # uncertainty is meaningful enough to guide anything
        X = rng.uniform(lo, hi, size=(self.n_init, dim))
        y = np.array([sign * self.func(x) for x in X])
        self.history_ = list(y * sign)

        for _ in range(self.n_iter):
            gp = GaussianProcessRegressor(length_scale=self.length_scale).fit(X, y)
            # optimise the acquisition by dense random search over the box --
            # cheap, since it only queries the GP, never the real objective
            cand = rng.uniform(lo, hi, size=(self.n_candidates, dim))
            mean, std = gp.predict(cand, return_std=True)
            scores = self._acquire(mean, std, y.min())
            x_next = cand[np.argmin(scores)]

            y_next = sign * self.func(x_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            self.history_.append(y_next * sign)

        best = np.argmin(y)
        self.best_x_ = X[best]
        self.best_y_ = float(y[best] * sign)
        self.X_ = X
        self.y_ = y * sign
        return self


__all__ = ["BayesianOptimization", "GaussianProcessRegressor",
           "expected_improvement", "upper_confidence_bound",
           "probability_of_improvement"]
