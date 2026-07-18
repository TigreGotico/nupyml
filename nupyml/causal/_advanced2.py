"""Causal inference v2: panel differences, heterogeneous effects, and boundary
identification.

Three more causal estimators. The Causal Forest estimates how a treatment effect
VARIES across individuals. Regression Discontinuity exploits a sharp
cutoff rule. The R-learner isolates the heterogeneous effect by residualising out
the confounded parts.
"""
import numpy as np

from ..base import BaseEstimator, clone
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor


class CausalForest(BaseEstimator):
    """How the treatment effect VARIES across people (Wager & Athey, 2018).

    An average treatment effect hides that a drug may help some patients and harm
    others. A causal forest estimates the CONDITIONAL effect ``tau(x)`` by growing
    many trees that split to separate regions of DIFFERENT effect (not different
    outcome), then, in each leaf, estimating the effect as treated-minus-control
    mean. "Honesty" -- using separate samples to choose splits and to estimate
    effects -- keeps the estimates unbiased. Averaging over trees gives a smooth,
    individualised ``tau(x)`` with far lower variance than a single tree. Assumes
    (approximately) randomised treatment.
    """

    def __init__(self, n_estimators=100, max_depth=4, min_leaf=5,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.random_state = random_state

    def _grow(self, X, t, y, depth, rng):
        node = {"tau": self._leaf_effect(t, y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            return node
        best = None
        feats = rng.choice(X.shape[1], max(1, int(np.sqrt(X.shape[1]))),
                           replace=False)
        for j in feats:
            for thr in np.quantile(X[:, j], [0.3, 0.5, 0.7]):
                left = X[:, j] <= thr
                if (left.sum() < self.min_leaf or (~left).sum() < self.min_leaf
                        or t[left].std() == 0 or t[~left].std() == 0):
                    continue
                tl = self._leaf_effect(t[left], y[left])
                tr = self._leaf_effect(t[~left], y[~left])
                het = left.sum() * (tl) ** 2 + (~left).sum() * (tr) ** 2  # heterogeneity
                if best is None or het > best[0]:
                    best = (het, j, thr, left)
        if best is None:
            return node
        _, j, thr, left = best
        node.update({"dim": j, "thr": thr,
                     "left": self._grow(X[left], t[left], y[left], depth + 1, rng),
                     "right": self._grow(X[~left], t[~left], y[~left], depth + 1, rng)})
        return node

    @staticmethod
    def _leaf_effect(t, y):
        if (t == 1).any() and (t == 0).any():
            return y[t == 1].mean() - y[t == 0].mean()
        return 0.0

    def fit(self, X, treatment, y):
        X = check_array(X)
        t = np.asarray(treatment); y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True)
            self.trees_.append(self._grow(X[idx], t[idx], y[idx], 0, rng))
        return self

    def _predict_tree(self, node, x):
        while "dim" in node:
            node = node["left"] if x[node["dim"]] <= node["thr"] else node["right"]
        return node["tau"]

    def predict(self, X):
        X = check_array(X)
        return np.array([np.mean([self._predict_tree(t, x) for t in self.trees_])
                         for x in X])


class RegressionDiscontinuity(BaseEstimator):
    """Identify an effect at a sharp CUTOFF rule (Thistlethwaite & Campbell, 1960).

    When treatment is assigned by a threshold -- a scholarship for scores >= 80, a
    subsidy for income <= a line -- units just above and just below the cutoff are
    essentially identical except for treatment. Regression discontinuity fits a
    LOCAL LINEAR regression on each side of the cutoff (within a bandwidth) and reads
    the treatment effect off the JUMP in the fitted line at the boundary. It needs no
    randomisation, only that everything else varies smoothly through the cutoff.
    ``running`` is the assignment variable, ``cutoff`` the threshold.
    """

    def __init__(self, cutoff=0.0, bandwidth=None):
        self.cutoff = cutoff
        self.bandwidth = bandwidth

    def fit(self, running, y):
        r = np.asarray(running, float).ravel() - self.cutoff
        y = np.asarray(y, float).ravel()
        h = self.bandwidth if self.bandwidth is not None else np.std(r)
        keep = np.abs(r) <= h
        r, y = r[keep], y[keep]
        left = r < 0
        # local linear fit on each side; the intercept gap at 0 is the effect
        al, bl = self._fit_line(r[left], y[left])
        ar, br = self._fit_line(r[~left], y[~left])
        self.effect_ = ar - al
        self.left_intercept_, self.right_intercept_ = al, ar
        return self

    @staticmethod
    def _fit_line(x, y):
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef[0], coef[1]

    def effect(self):
        return self.effect_


class RLearner(BaseEstimator):
    """Isolate the heterogeneous effect by RESIDUALISING (Nie & Wager, 2021).

    Confounding contaminates a naive effect estimate: outcome and treatment both
    depend on covariates. The R-learner removes that in two stages. First fit the
    "nuisances" -- ``m(x) = E[Y|X]`` and the propensity ``e(x) = E[T|X]`` -- and take
    RESIDUALS ``Y - m(x)`` and ``T - e(x)``. Robinson's transformation shows the
    heterogeneous effect ``tau(x)`` is exactly the regression of the outcome residual
    on the treatment residual, so a weighted least-squares on those residuals
    recovers ``tau`` free of the confounding both stages absorbed. Linear ``tau(x)``
    here.
    """

    def __init__(self, outcome_model=None, propensity_model=None):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model

    def fit(self, X, treatment, y):
        X = check_array(X)
        t = np.asarray(treatment, float); y = np.asarray(y, float)
        om = clone(self.outcome_model) if self.outcome_model is not None \
            else DecisionTreeRegressor(max_depth=4)
        pm = clone(self.propensity_model) if self.propensity_model is not None \
            else DecisionTreeRegressor(max_depth=4)
        m_hat = om.fit(X, y).predict(X)                   # E[Y|X]
        e_hat = np.clip(pm.fit(X, t).predict(X), 0.02, 0.98)   # E[T|X]
        y_res = y - m_hat
        t_res = t - e_hat
        # Robinson: minimise sum (y_res - t_res * X @ theta)^2  => weighted LS
        Xd = np.column_stack([np.ones(len(y)), X]) * t_res[:, None]
        theta, *_ = np.linalg.lstsq(Xd, y_res, rcond=None)
        self.theta_ = theta
        self._m, self._e = om, pm
        return self

    def predict(self, X):
        X = check_array(X)
        return np.column_stack([np.ones(len(X)), X]) @ self.theta_


__all__ = ["CausalForest", "RegressionDiscontinuity", "RLearner"]
