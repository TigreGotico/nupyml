"""Generalized additive models: one smooth curve per feature, added up.

THE MODEL
---------
::

    y = b0 + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)

Compare the two things it sits between::

    linear:  y = b0 + b_1 x_1 + ... + b_p x_p     each effect is a STRAIGHT LINE
    GAM:     y = b0 + f_1(x_1) + ... + f_p(x_p)   each effect is ANY smooth curve
    general: y = f(x_1, ..., x_p)                 anything at all

The middle one keeps the structure of the first while relaxing its shape
constraint. Each feature still gets its own separate, isolated contribution --
the effects merely stop being lines.

WHY THAT STRUCTURE IS WORTH KEEPING
-----------------------------------
Two reasons, and they are the whole case for GAMs:

* **It defeats the curse of dimensionality.** Estimating a general ``f`` in 10
  dimensions requires data exponential in 10. Estimating 10 separate ONE-
  dimensional functions requires ten one-dimensional problems, and they are
  easy. The additive assumption converts an impossible problem into ten trivial
  ones.
* **It stays interpretable, and interpretable in the strongest sense.** You can
  PLOT ``f_j``. That plot is not an approximation of the model, a summary, or a
  post-hoc explanation like SHAP -- it IS the model's treatment of feature ``j``,
  exactly and completely. Nothing else in this library offers that above
  linearity.

WHAT IT GIVES UP
----------------
Interactions. A GAM cannot represent "the effect of dose depends on age" --
that is a function of two variables, and by construction there are none.
Where interactions genuinely drive the outcome, a GAM will underfit and no
amount of smoothing will save it. ``GA2M`` (adding selected pairwise terms) is
the standard extension, trading some of the interpretability back for it.

BACKFITTING: THE ALGORITHM
--------------------------
How do you fit ten functions that must sum to ``y``? One at a time, cycling::

    repeat until nothing changes:
        for each feature j:
            partial residual = y - b0 - (sum of all the OTHER f_k)
            refit f_j against x_j on that residual

The trick is the PARTIAL RESIDUAL: what ``y`` still has left to explain once
everyone else has spoken. Fitting ``f_j`` to it is a one-dimensional smoothing
problem, which we can already solve. So an intractable joint fit becomes a loop
over easy fits -- which is Gauss-Seidel, the same idea as coordinate descent in
``Lasso``.

Each ``f_j`` is centred to mean zero after every update. Without that the
intercept is unidentifiable: adding a constant to ``f_1`` and subtracting it from
``f_2`` gives an identical model, so the fit could drift forever without the
predictions changing at all.

Hastie & Tibshirani (1986).
"""
import numpy as np

from ..base import (BaseEstimator, RegressorMixin, ClassifierMixin,
                    TransformerMixin, check_is_fitted)
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, sigmoid


def natural_cubic_basis(x, knots):
    """Natural cubic spline basis: piecewise cubics, joined smoothly, LINEAR at
    the ends.

    "Natural" means the second derivative is forced to zero beyond the boundary
    knots, which makes the fit linear out there. That constraint sounds arbitrary
    and is deeply practical: an unconstrained cubic spline does whatever a cubic
    does past its last knot, which is to shoot off toward infinity. Exactly where
    the data is thinnest and the fit least certain, the model would otherwise be
    at its most confident and most absurd. Forcing linearity is the modest
    behaviour you would want anyway.

    Returns a design matrix with ``len(knots) - 1`` columns.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64)
    K = len(knots)
    if K < 3:
        raise ValueError(f"need at least 3 knots, got {K}")

    def d(k):
        # the standard construction; the ((x-k)_+)^3 pieces are what make it
        # cubic between knots, and differencing them is what makes it natural
        num = np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3
        return num / (knots[-1] - knots[k])

    basis = [x]
    for k in range(K - 2):
        basis.append(d(k) - d(K - 2))
    return np.column_stack(basis)


class SplineTransformer(BaseEstimator, TransformerMixin):
    """Expand each feature into a spline basis.

    This is the "just make the features non-linear and keep the linear model"
    move, done properly. A polynomial basis (``x``, ``x^2``, ``x^3``) is the
    naive version and is bad for a reason worth knowing: polynomial terms are
    GLOBAL, so a wiggle needed at one end of the range changes the fit at the
    other end, and high-degree polynomials oscillate wildly near the boundaries
    (Runge's phenomenon).

    A spline basis is LOCAL. Each basis function is non-zero only near its knot,
    so fitting a bump at ``x=3`` leaves ``x=30`` alone. Same trick -- expand into
    more columns and stay linear in them -- without the pathology.

    ``knots="quantile"`` puts knots at data quantiles, so flexibility follows the
    data rather than the axis: where there are many points there is detail to
    resolve, and where there are few there is nothing to fit.
    """

    def __init__(self, n_knots=5, knots="quantile", include_bias=False):
        self.n_knots = n_knots
        self.knots = knots
        self.include_bias = include_bias

    def _knots_for(self, col):
        if self.knots == "quantile":
            return np.unique(np.percentile(col, np.linspace(0, 100, self.n_knots)))
        if self.knots == "uniform":
            return np.linspace(col.min(), col.max(), self.n_knots)
        raise ValueError(f"Unknown knots strategy: {self.knots!r}")

    def fit(self, X, y=None):
        X = check_array(X)
        self.knots_ = [self._knots_for(X[:, j]) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(max(len(k) - 1, 1) for k in self.knots_) \
            + int(self.include_bias)
        return self

    def transform(self, X):
        check_is_fitted(self, "knots_")
        X = check_array(X)
        blocks = []
        if self.include_bias:
            blocks.append(np.ones((len(X), 1)))
        for j, knots in enumerate(self.knots_):
            if len(knots) < 3:
                # a near-constant feature has no interior structure to model
                blocks.append(X[:, [j]])
            else:
                blocks.append(natural_cubic_basis(X[:, j], knots))
        return np.hstack(blocks)


class _SmoothTerm:
    """One feature's contribution: a penalised spline fit against a residual."""

    def __init__(self, x, n_knots, lam):
        self.knots = np.unique(np.percentile(x, np.linspace(0, 100, n_knots)))
        self.lam = lam
        self.usable = len(self.knots) >= 3
        self.beta = None

    def basis(self, x):
        if not self.usable:
            return np.asarray(x).reshape(-1, 1)
        return natural_cubic_basis(x, self.knots)

    def fit(self, x, residual):
        B = self.basis(x)
        # ridge on the basis coefficients: a spline with many knots interpolates
        # the noise, so the penalty -- not the knot count -- is what controls
        # smoothness. This is the cheap stand-in for a proper second-derivative
        # penalty, and it behaves the same way: lam -> inf gives a straight line
        A = B.T @ B + self.lam * np.eye(B.shape[1])
        self.beta = np.linalg.solve(A, B.T @ residual)
        out = B @ self.beta
        # centred, or the intercept is unidentifiable -- see the module docstring
        self.mean = out.mean()
        self.beta_intercept = self.mean
        return out - self.mean

    def predict(self, x):
        return self.basis(x) @ self.beta - self.mean


class GAM(BaseEstimator, RegressorMixin):
    """Additive model fitted by backfitting.

    ``partial_dependence(j)`` returns the fitted ``f_j`` -- plot it and you have
    read the model. That is the reason to use this over a forest.
    """

    def __init__(self, n_knots=8, lam=1.0, max_iter=50, tol=1e-5):
        self.n_knots = n_knots
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, p = X.shape
        self.n_features_in_ = p

        self.intercept_ = float(y.mean())
        self.terms_ = [_SmoothTerm(X[:, j], self.n_knots, self.lam)
                       for j in range(p)]
        # each feature's current contribution, all starting at "nothing"
        contributions = np.zeros((n, p))
        self.n_iter_ = 0

        for it in range(self.max_iter):
            previous = contributions.copy()
            for j in range(p):
                # the partial residual: what y has left once everyone ELSE has
                # had their say. This is the whole trick -- see the module
                # docstring
                others = contributions.sum(axis=1) - contributions[:, j]
                partial = y - self.intercept_ - others
                contributions[:, j] = self.terms_[j].fit(X[:, j], partial)

            self.n_iter_ = it + 1
            change = np.abs(contributions - previous).mean()
            if change < self.tol:
                break

        self.contributions_ = contributions
        return self

    def predict(self, X):
        check_is_fitted(self, "terms_")
        X = check_array(X)
        out = np.full(len(X), self.intercept_)
        for j, term in enumerate(self.terms_):
            out += term.predict(X[:, j])
        return out

    def partial_dependence(self, feature, grid=None, n_points=100):
        """The fitted ``f_j`` over a grid. THIS IS THE MODEL, for feature j.

        Not an approximation of it, not a post-hoc attribution like SHAP or LIME
        -- the actual function the model adds for this feature, read off exactly.
        Nothing above a linear model in this library can say that.
        """
        check_is_fitted(self, "terms_")
        term = self.terms_[feature]
        if grid is None:
            grid = np.linspace(term.knots[0], term.knots[-1], n_points)
        return np.asarray(grid), term.predict(np.asarray(grid))


class GAMClassifier(BaseEstimator, ClassifierMixin):
    """A GAM for binary classification, by backfitting on the logit scale.

    Uses the local-scoring algorithm: repeatedly form a working response (the
    linearised logit) and its weights, then backfit against that -- exactly the
    IRLS trick that turns logistic regression into repeated weighted least
    squares, with a smoother in place of the linear fit.
    """

    def __init__(self, n_knots=8, lam=1.0, max_iter=25, tol=1e-5):
        self.n_knots = n_knots
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        if len(self.classes_) != 2:
            raise ValueError("GAMClassifier supports binary targets only")
        t = self._le.transform(y).astype(np.float64)

        n, p = X.shape
        self.n_features_in_ = p
        prior = np.clip(t.mean(), 1e-6, 1 - 1e-6)
        self.intercept_ = float(np.log(prior / (1 - prior)))
        self.terms_ = [_SmoothTerm(X[:, j], self.n_knots, self.lam)
                       for j in range(p)]
        contributions = np.zeros((n, p))

        for _ in range(self.max_iter):
            eta = self.intercept_ + contributions.sum(axis=1)
            mu = sigmoid(eta)
            w = np.clip(mu * (1 - mu), 1e-6, None)
            # the working response: where the linearised logit says eta should
            # move. Smoothing THIS is what makes it a GAM rather than a GLM
            z = eta + (t - mu) / w

            previous = contributions.copy()
            for j in range(p):
                others = contributions.sum(axis=1) - contributions[:, j]
                partial = z - self.intercept_ - others
                contributions[:, j] = self.terms_[j].fit(X[:, j], partial)
            if np.abs(contributions - previous).mean() < self.tol:
                break

        return self

    def decision_function(self, X):
        check_is_fitted(self, "terms_")
        X = check_array(X)
        out = np.full(len(X), self.intercept_)
        for j, term in enumerate(self.terms_):
            out += term.predict(X[:, j])
        return out

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def partial_dependence(self, feature, grid=None, n_points=100):
        """The fitted ``f_j`` on the LOG-ODDS scale."""
        check_is_fitted(self, "terms_")
        term = self.terms_[feature]
        if grid is None:
            grid = np.linspace(term.knots[0], term.knots[-1], n_points)
        return np.asarray(grid), term.predict(np.asarray(grid))


__all__ = ["GAM", "GAMClassifier", "SplineTransformer", "natural_cubic_basis"]
