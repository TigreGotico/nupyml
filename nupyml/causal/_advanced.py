"""Causal inference and discovery: IV, double ML, CATE learners, panel methods,
and DAG structure learning.

The base ``causal`` module estimates treatment effects by adjusting for observed
confounders (IPW, doubly-robust). These add the tools for UNOBSERVED confounding
(instruments), machine-learning nuisance models (double ML), HETEROGENEOUS
effects (meta-learners), PANEL data (diff-in-diff, synthetic control), and
discovering the causal graph itself (NOTEARS).
"""
import numpy as np

from ..base import BaseEstimator, clone
from ..utils import check_array


class InstrumentalVariables(BaseEstimator):
    """Two-stage least squares -- identify an effect despite UNOBSERVED confounding.

    THE PROBLEM ADJUSTMENT CANNOT FIX
    ---------------------------------
    IPW and regression only remove confounding you can MEASURE. If an unobserved
    variable drives both the treatment and the outcome (ability drives both
    schooling and wages), no amount of adjusting for observed covariates recovers
    the causal effect. An INSTRUMENT ``Z`` breaks the deadlock: it moves the
    treatment but affects the outcome ONLY through the treatment (distance to
    college shifts schooling but not wages directly).

    2SLS uses it in two regressions: (1) regress treatment on the instrument to
    get the part of treatment the instrument EXPLAINS (confounder-free by
    assumption), then (2) regress the outcome on THAT predicted treatment. The
    second stage's slope is the causal effect. ``fit(Z, T, y)``; the effect is in
    ``coef_``.
    """

    def fit(self, Z, T, y):
        Z = check_array(Z)
        T = np.asarray(T, float)
        y = np.asarray(y, float)
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        # stage 1: project treatment onto the instrument(s)
        beta1, *_ = np.linalg.lstsq(Z1, T, rcond=None)
        T_hat = Z1 @ beta1
        # stage 2: regress outcome on the fitted (confounder-free) treatment
        D = np.column_stack([np.ones(len(y)), T_hat])
        beta2, *_ = np.linalg.lstsq(D, y, rcond=None)
        self.intercept_, self.coef_ = beta2[0], beta2[1]
        return self


class DoubleML(BaseEstimator):
    """Double / debiased machine learning for the average treatment effect
    (Chernozhukov et al., 2018).

    THE ORTHOGONALISATION
    ---------------------
    You want to control for confounders with flexible ML models, but plugging ML
    predictions straight in biases the effect (regularisation bleaks into the
    estimate). Double ML removes this: fit two nuisance models -- outcome ``E[y|X]``
    and treatment ``E[T|X]`` -- then regress the RESIDUALS of ``y`` on the
    RESIDUALS of ``T``. Partialling both sides out (Frisch-Waugh-Lovell) makes the
    estimate ORTHOGONAL to small nuisance errors, so ML-grade confounder models are
    safe to use. CROSS-FITTING (predict each fold from the others) removes the
    remaining overfitting bias. ``effect_`` is the estimated ATE.
    """

    def __init__(self, outcome_model=None, treatment_model=None, cv=2,
                 random_state=None):
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        from ..model_selection import KFold
        X = check_array(X)
        T = np.asarray(T, float)
        y = np.asarray(y, float)
        om = self.outcome_model or RandomForestRegressor(n_estimators=50,
                                                         random_state=0)
        tm = self.treatment_model or RandomForestRegressor(n_estimators=50,
                                                          random_state=0)
        y_res = np.zeros_like(y)
        t_res = np.zeros_like(T)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train, test in kf.split(X):
            # cross-fitting: nuisance predictions from the OTHER fold
            y_res[test] = y[test] - clone(om).fit(X[train], y[train]).predict(X[test])
            t_res[test] = T[test] - clone(tm).fit(X[train], T[train]).predict(X[test])
        # effect = slope of residual-y on residual-T
        self.effect_ = float(np.sum(t_res * y_res) / (np.sum(t_res ** 2) + 1e-12))
        return self


class TLearner(BaseEstimator):
    """CATE by fitting SEPARATE outcome models per treatment arm.

    Heterogeneous treatment effects ask how the effect VARIES across individuals.
    The T-learner fits one regressor on the treated, another on the controls, and
    estimates each unit's effect as the difference of the two predictions::

        tau(x) = mu_treated(x) - mu_control(x)

    Simple and flexible, but it splits the data and can't share strength between
    arms -- the X-learner refines exactly that. ``predict_effect(X)`` returns the
    per-unit CATE.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        X = check_array(X); T = np.asarray(T); y = np.asarray(y, float)
        base = self.base_estimator or RandomForestRegressor(n_estimators=100,
                                                            random_state=0)
        self.m1_ = clone(base).fit(X[T == 1], y[T == 1])
        self.m0_ = clone(base).fit(X[T == 0], y[T == 0])
        return self

    def predict_effect(self, X):
        X = check_array(X)
        return self.m1_.predict(X) - self.m0_.predict(X)


class XLearner(BaseEstimator):
    """CATE that IMPUTES individual effects then models them (Künzel et al., 2019).

    The T-learner wastes information when the arms are imbalanced. The X-learner
    adds a clever middle step: use each arm's model to IMPUTE the counterfactual
    for the OTHER arm, giving a pseudo-effect per unit, then fit a model to those
    pseudo-effects in each arm and blend the two by the propensity score. This
    borrows strength across arms and shines when one group is much larger -- the
    common case in observational data. ``predict_effect(X)`` returns the CATE.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        from ..linear_model import LogisticRegression
        X = check_array(X); T = np.asarray(T); y = np.asarray(y, float)
        base = self.base_estimator or RandomForestRegressor(n_estimators=100,
                                                            random_state=0)
        m1 = clone(base).fit(X[T == 1], y[T == 1])
        m0 = clone(base).fit(X[T == 0], y[T == 0])
        # imputed treatment effects within each arm
        d1 = y[T == 1] - m0.predict(X[T == 1])
        d0 = m1.predict(X[T == 0]) - y[T == 0]
        self.tau1_ = clone(base).fit(X[T == 1], d1)
        self.tau0_ = clone(base).fit(X[T == 0], d0)
        self.prop_ = LogisticRegression(max_iter=300).fit(X, T)
        return self

    def predict_effect(self, X):
        X = check_array(X)
        g = self.prop_.predict_proba(X)[:, 1]         # propensity as the blend weight
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)


def difference_in_differences(y_control_pre, y_control_post,
                              y_treated_pre, y_treated_post):
    """The DiD estimator: subtract the control group's TREND from the treated's.

    With before/after data on a treated and a control group, the treated group's
    raw change confounds the treatment with whatever else happened over time. DiD
    removes the common trend by taking the DIFFERENCE OF DIFFERENCES::

        effect = (treated_post - treated_pre) - (control_post - control_pre)

    Its one assumption is PARALLEL TRENDS: absent treatment, the two groups would
    have moved together. Then the control's change estimates what the treated
    would have done anyway, and the excess is the causal effect. Accepts group
    means (scalars) or arrays (averaged).
    """
    def m(v):
        return float(np.mean(v))
    return (m(y_treated_post) - m(y_treated_pre)) - (m(y_control_post) - m(y_control_pre))


class SyntheticControl(BaseEstimator):
    """Build a weighted combination of controls that mimics the treated unit
    (Abadie & Gardeazabal, 2003).

    When there is ONE treated unit (a country, a state) and several controls, no
    single control is a good counterfactual. Synthetic control finds convex
    weights over the controls so their weighted PRE-treatment outcomes track the
    treated unit's; that "synthetic" unit's POST-treatment path is the
    counterfactual, and the gap from the real treated path is the effect. The
    weights are non-negative and sum to one, so the synthetic unit stays an
    interpolation of real units (no extrapolation). ``fit`` on pre-period matrices,
    ``effect`` on the post period.
    """

    def __init__(self, max_iter=5000, lr=0.01):
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, treated_pre, controls_pre):
        # controls_pre: (T_pre, n_controls); treated_pre: (T_pre,)
        Y = check_array(controls_pre)
        t = np.asarray(treated_pre, float)
        n = Y.shape[1]
        w = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):                # projected gradient on the simplex
            grad = Y.T @ (Y @ w - t)
            w = w - self.lr * grad / len(t)
            w = np.maximum(w, 0)
            s = w.sum()
            w = w / s if s > 0 else np.full(n, 1.0 / n)
        self.weights_ = w
        return self

    def effect(self, treated_post, controls_post):
        synthetic = check_array(controls_post) @ self.weights_
        return np.asarray(treated_post, float) - synthetic


def notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16):
    """NOTEARS: learn a causal DAG by CONTINUOUS optimization (Zheng et al., 2018).

    THE BREAKTHROUGH
    ----------------
    Learning a DAG was a combinatorial nightmare -- searching over acyclic
    structures is NP-hard. NOTEARS turned it into smooth optimization with one
    trick: a differentiable measure of "how cyclic" a weighted adjacency ``W`` is,
    ``h(W) = tr(e^{W∘W}) - d``, which is EXACTLY zero iff the graph is acyclic. So
    you minimise the reconstruction loss ``||X - XW||^2 + lambda1||W||_1`` subject
    to ``h(W)=0``, enforced by an augmented-Lagrangian outer loop. The result is a
    weighted DAG learned by gradient descent -- no discrete search.

    Returns the weighted adjacency ``W`` (``W[i,j] != 0`` means i -> j).
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize
    X = check_array(X)
    n, d = X.shape

    def loss(w):
        W = w.reshape(d, d)
        R = X - X @ W
        f = 0.5 / n * (R ** 2).sum()
        G = -1.0 / n * X.T @ R
        return f, G.flatten()

    def h_func(W):
        E = expm(W * W)
        return np.trace(E) - d, E.T * W * 2           # value and gradient

    rho, alpha, w = 1.0, 0.0, np.zeros(d * d)
    for _ in range(max_iter):
        def obj(w):
            W = w.reshape(d, d)
            f, Gf = loss(w)
            h, Gh = h_func(W)
            total = f + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(w).sum()
            grad = Gf + (rho * h + alpha) * Gh.flatten() + lambda1 * np.sign(w)
            return total, grad
        sol = minimize(obj, w, jac=True, method="L-BFGS-B")
        w = sol.x
        h_val = h_func(w.reshape(d, d))[0]
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break
        rho *= 10
    W = w.reshape(d, d)
    W[np.abs(W) < 0.3] = 0                            # threshold weak edges
    return W


__all__ = ["InstrumentalVariables", "DoubleML", "TLearner", "XLearner",
           "difference_in_differences", "SyntheticControl", "notears_linear"]
