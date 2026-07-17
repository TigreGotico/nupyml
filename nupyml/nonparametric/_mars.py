"""MARS: find the bends automatically, and write them down as hinges.

THE BUILDING BLOCK
------------------
A HINGE function -- ``max(0, x - k)`` or ``max(0, k - x)``. It is zero on one
side of the knot ``k`` and linear on the other: a line with an elbow. Add a few
and you get a piecewise-linear function; add them across several features and you
get a flexible regression surface.

The appeal over splines is that MARS DISCOVERS the knots. A spline needs you to
place them; MARS searches for where the relationship actually bends and puts a
hinge there. The output is a short, readable formula::

    y = 25 + 4*max(0, x1 - 3) - 2*max(0, 3 - x1) + 5*max(0, x2 - 7)

which a domain expert can check against what they already believe -- unlike a
forest, which offers nothing to check.

THE TWO PASSES
--------------
Directly from CART, and the resemblance is not coincidental -- Friedman invented
both:

* **Forward pass -- greedily overfit.** Repeatedly add the hinge pair (a knot on
  some feature, both directions) that most reduces training error. Hinges may
  MULTIPLY existing terms, which is how MARS gets interactions -- a product of
  two hinges is a function of two variables, the thing a GAM structurally cannot
  represent. This keeps going well past the right model, on purpose.
* **Backward pass -- prune.** Delete terms one at a time, each time removing
  whichever hurts least, and keep the sub-model that scores best on GCV.

Overfit then prune is the same philosophy as growing a full tree and pruning it
back: it is far easier to recognise a good model by removing from a rich one than
to build it up perfectly in a single greedy sweep, because a term that looks
useless alone may be essential in combination.

GCV: PRUNING WITHOUT A VALIDATION SET
-------------------------------------
Generalized Cross-Validation estimates out-of-sample error from the training fit
alone::

    GCV = MSE / (1 - (effective_params) / n)^2

The denominator is the penalty: more terms shrink it, inflating GCV, so adding a
term must EARN its place by reducing MSE more than it costs in complexity. The
"effective parameters" count the knots too -- and each knot costs more than a
plain coefficient (``penalty`` per knot, ~3 by default), because a knot was
CHOSEN by looking at the data, and a chosen parameter overfits more than a fixed
one. That is a genuine statistical insight, not a fudge factor: the search itself
is a source of overfitting, and GCV prices it in.

Friedman (1991).
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


class _HingeTerm:
    """A product of hinge functions -- one basis term of the MARS model.

    A term is a list of ``(feature, knot, direction)`` factors, multiplied. An
    empty list is the intercept (the constant 1).
    """

    def __init__(self, factors=None):
        self.factors = factors or []

    def evaluate(self, X):
        out = np.ones(len(X))
        for feature, knot, direction in self.factors:
            if direction > 0:
                out = out * np.maximum(0.0, X[:, feature] - knot)
            else:
                out = out * np.maximum(0.0, knot - X[:, feature])
        return out

    def __str__(self):
        if not self.factors:
            return "1"
        parts = []
        for feature, knot, direction in self.factors:
            if direction > 0:
                parts.append(f"max(0, x{feature} - {knot:.3g})")
            else:
                parts.append(f"max(0, {knot:.3g} - x{feature})")
        return " * ".join(parts)


class MARS(BaseEstimator, RegressorMixin):
    """Multivariate adaptive regression splines.

    ``max_terms`` caps the forward pass; ``max_degree`` caps how many hinges may
    multiply (so ``max_degree=1`` forbids interactions and gives a
    piecewise-linear additive model -- a GAM with learned knots).
    """

    def __init__(self, max_terms=21, max_degree=2, penalty=3.0,
                 min_samples_split=10):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.min_samples_split = min_samples_split

    def _fit_coef(self, terms, X, y):
        """Least squares of ``y`` on the current basis. Returns (coef, sse)."""
        B = np.column_stack([t.evaluate(X) for t in terms])
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ coef
        return coef, float(resid @ resid), B

    def _gcv(self, sse, n, n_terms):
        """Training error, penalised for the effective number of parameters.

        Each term is one parameter; each knot adds ``penalty`` more, because a
        knot chosen by searching the data overfits more than a fixed coefficient
        -- see the module docstring.
        """
        # n_terms coefficients + (n_terms - 1) knots, knots at `penalty` each
        effective = n_terms + self.penalty * (n_terms - 1)
        denom = (1.0 - effective / n) ** 2
        if denom <= 0:
            return np.inf
        return (sse / n) / denom

    def _forward_pass(self, X, y):
        n, p = X.shape
        terms = [_HingeTerm([])]        # start with just the intercept

        while len(terms) + 2 <= self.max_terms:
            best_gain, best_pair = -np.inf, None
            _, base_sse, _ = self._fit_coef(terms, X, y)

            for parent in terms:
                if len(parent.factors) >= self.max_degree:
                    continue            # do not exceed the interaction limit
                used = {f[0] for f in parent.factors}
                parent_vals = parent.evaluate(X)
                active = parent_vals > 0     # a knot only helps where the parent
                if active.sum() < self.min_samples_split:   # term is non-zero
                    continue

                for feature in range(p):
                    if feature in used:
                        continue        # no two hinges on one feature in a term
                    # candidate knots: the observed values where the parent is
                    # active. Every distinct value is a candidate, but scoring
                    # them all is the expensive part, so subsample if there are
                    # very many
                    candidates = np.unique(X[active, feature])
                    if len(candidates) > 50:
                        candidates = np.quantile(candidates,
                                                 np.linspace(0, 1, 50))
                    for knot in candidates:
                        trial = terms + [
                            _HingeTerm(parent.factors + [(feature, knot, 1)]),
                            _HingeTerm(parent.factors + [(feature, knot, -1)])]
                        try:
                            _, sse, _ = self._fit_coef(trial, X, y)
                        except np.linalg.LinAlgError:
                            continue
                        gain = base_sse - sse
                        if gain > best_gain:
                            best_gain, best_pair = gain, (parent, feature, knot)

            if best_pair is None or best_gain <= 1e-12:
                break
            parent, feature, knot = best_pair
            # both directions are added as a pair, so the model can bend either
            # way at this knot -- the "reflected pair" that gives MARS its shape
            terms.append(_HingeTerm(parent.factors + [(feature, knot, 1)]))
            terms.append(_HingeTerm(parent.factors + [(feature, knot, -1)]))
        return terms

    def _backward_pass(self, terms, X, y):
        n = len(y)
        _, sse, _ = self._fit_coef(terms, X, y)
        best_terms = list(terms)
        best_gcv = self._gcv(sse, n, len(terms))

        current = list(terms)
        while len(current) > 1:
            # try deleting each non-intercept term; keep the deletion that
            # leaves the lowest GCV
            best_delete, best_delete_gcv = None, np.inf
            for i in range(1, len(current)):
                trial = current[:i] + current[i + 1:]
                try:
                    _, trial_sse, _ = self._fit_coef(trial, X, y)
                except np.linalg.LinAlgError:
                    continue
                gcv = self._gcv(trial_sse, n, len(trial))
                if gcv < best_delete_gcv:
                    best_delete_gcv, best_delete = gcv, trial
            if best_delete is None:
                break
            current = best_delete
            if best_delete_gcv < best_gcv:
                best_gcv, best_terms = best_delete_gcv, list(current)
        return best_terms, best_gcv

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        terms = self._forward_pass(X, y)
        terms, self.gcv_ = self._backward_pass(terms, X, y)

        self.terms_ = terms
        self.coef_, self.sse_, _ = self._fit_coef(terms, X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "terms_")
        X = check_array(X)
        B = np.column_stack([t.evaluate(X) for t in self.terms_])
        return B @ self.coef_

    @property
    def expression_(self):
        """The fitted model as a readable formula -- the reason to use MARS."""
        check_is_fitted(self, "terms_")
        parts = [f"{self.coef_[0]:.4g}"]
        for c, t in zip(self.coef_[1:], self.terms_[1:]):
            parts.append(f"{c:+.4g} * {t}")
        return " ".join(parts)


__all__ = ["MARS"]
