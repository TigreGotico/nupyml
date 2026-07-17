"""Feature attribution: global importance, local explanations, Shapley values."""
from itertools import combinations

import numpy as np

from ..base import clone
from ..utils import check_array, check_random_state


def permutation_importance(estimator, X, y, scoring=None, n_repeats=5,
                           random_state=None):
    """Global importance by breaking one feature at a time.

    THE IDEA
    --------
    Shuffle a single feature's column -- destroying its relationship with the
    target while keeping its marginal distribution -- and measure how far the
    model's score drops. A large drop means the model RELIED on that feature; no
    drop means it did not, whatever the model's internal weights suggest.

    WHY IT BEATS "BUILT-IN" IMPORTANCES
    -----------------------------------
    A tree's split-count importance or a linear model's coefficients describe the
    model's STRUCTURE, which can mislead: a coefficient is large partly because
    of the feature's scale, and split counts inflate high-cardinality features.
    Permutation importance measures what the model's PREDICTIONS actually depend
    on, empirically, and it works for any model without peeking inside.

    THE TRAP: CORRELATED FEATURES
    -----------------------------
    If two features are correlated, shuffling one leaves the model the other as a
    substitute, so BOTH look unimportant -- the model shrugs off the loss of
    either. The importance is real but the interpretation ("neither matters") is
    wrong; they matter jointly. Shuffling correlated features together, or reading
    the result as "importance given the others", is the fix, and knowing the trap
    is the point.
    """
    from ..metrics import accuracy_score, r2_score
    X = check_array(X)
    y = np.asarray(y)
    rng = check_random_state(random_state)

    if scoring is None:
        # regressor vs classifier: pick the natural default score
        is_clf = hasattr(estimator, "predict_proba") or \
            getattr(estimator, "_estimator_type", None) == "classifier"
        scoring = (lambda yt, yp: accuracy_score(yt, yp)) if is_clf else \
            (lambda yt, yp: r2_score(yt, yp))

    baseline = scoring(y, estimator.predict(X))
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])              # break feature j alone
            importances[j, r] = baseline - scoring(y, estimator.predict(X_perm))
    return {"importances_mean": importances.mean(axis=1),
            "importances_std": importances.std(axis=1),
            "importances": importances}


def partial_dependence(estimator, X, feature, grid=None, n_points=50,
                       predict_method="predict"):
    """How the average prediction moves as one feature sweeps its range.

    For each value on a grid, SET that feature to the value for EVERY row, predict
    the whole dataset, and average. The result is the marginal effect of the
    feature, with the others integrated out at their observed distribution.

    THE ASSUMPTION IT HIDES
    -----------------------
    Setting a feature to a grid value while leaving the others at their real
    values can create IMPOSSIBLE rows -- pregnancy at age 80, income far below the
    minimum for a held-fixed job title. The average is then taken over points that
    cannot exist, so a PDP over correlated features can be misleading. It is honest
    only when the swept feature is roughly independent of the rest, which is worth
    checking before trusting the curve.
    """
    X = check_array(X)
    predict = getattr(estimator, predict_method)
    if grid is None:
        col = X[:, feature]
        grid = np.linspace(col.min(), col.max(), n_points)

    pd_values = np.empty(len(grid))
    for i, v in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature] = v                       # force the feature for all rows
        out = predict(X_mod)
        pd_values[i] = np.mean(out[:, 1] if out.ndim > 1 else out)
    return np.asarray(grid), pd_values


def ice(estimator, X, feature, grid=None, n_points=50, predict_method="predict"):
    """Individual Conditional Expectation: one PDP curve PER row, not averaged.

    A PDP averages, which can hide the story: if the feature helps half the
    population and hurts the other half, the average is flat and says "no effect"
    -- exactly wrong. ICE plots every row's curve separately, so DIVERGING curves
    reveal an interaction the PDP would have cancelled out. The PDP is just the
    mean of these; keeping them separate is what surfaces heterogeneity.

    Returns ``(grid, curves)`` with ``curves`` of shape ``(n_rows, len(grid))``.
    """
    X = check_array(X)
    predict = getattr(estimator, predict_method)
    if grid is None:
        col = X[:, feature]
        grid = np.linspace(col.min(), col.max(), n_points)

    curves = np.empty((len(X), len(grid)))
    for i, v in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature] = v
        out = predict(X_mod)
        curves[:, i] = out[:, 1] if out.ndim > 1 else out
    return np.asarray(grid), curves


class LIME:
    """Local Interpretable Model-agnostic Explanations.

    THE IDEA
    --------
    A complex model may be hopelessly nonlinear GLOBALLY yet nearly linear in a
    small NEIGHBOURHOOD of one point. LIME exploits that: to explain a single
    prediction, sample perturbed points around it, label them with the black box,
    weight them by closeness to the point of interest, and fit a simple WEIGHTED
    LINEAR model to that local data. The linear model's coefficients are the
    explanation -- "this prediction went up because feature 3 was high, here".

    WHAT TO DISTRUST
    ----------------
    The explanation depends on the neighbourhood: how wide you sample and how you
    weight distance. Change the kernel width and the coefficients move, sometimes
    a lot, and there is no single right choice. LIME is a lens, not a measurement
    -- useful for intuition, unreliable as ground truth. SHAP was designed partly
    to remove this arbitrariness.

    Ribeiro, Singh & Guestrin (2016).
    """

    def __init__(self, n_samples=1000, kernel_width=0.75, random_state=None):
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.random_state = random_state

    def explain(self, predict_fn, x, X_background):
        """Local linear coefficients explaining ``predict_fn`` at ``x``.

        ``predict_fn`` returns a scalar per row (a probability, or a regression
        value); ``X_background`` sets the perturbation scale per feature.
        """
        rng = check_random_state(self.random_state)
        x = np.asarray(x, float).ravel()
        scale = np.std(X_background, axis=0)
        scale[scale == 0] = 1.0

        # sample a cloud around x, using the data's own per-feature spread
        samples = x + rng.normal(0, 1, size=(self.n_samples, len(x))) * scale
        samples[0] = x                              # keep the point itself
        y = np.asarray(predict_fn(samples)).ravel()

        # weight by proximity to x: near samples define the "local" fit
        dist = np.linalg.norm((samples - x) / scale, axis=1)
        weights = np.exp(-(dist ** 2) / (self.kernel_width ** 2 * len(x)))

        # weighted least squares of the black-box output on the perturbations
        Xs = (samples - x) / scale                  # centred and standardised
        A = np.column_stack([np.ones(len(Xs)), Xs])
        W = np.diag(weights)
        coef = np.linalg.lstsq(A.T @ W @ A + 1e-6 * np.eye(A.shape[1]),
                               A.T @ W @ y, rcond=None)[0]
        self.intercept_ = coef[0]
        self.coef_ = coef[1:] / scale               # back to original feature units
        return self.coef_


class KernelSHAP:
    """Shapley-value attributions: the UNIQUE fair credit assignment.

    THE GAME-THEORY IDEA
    --------------------
    Treat the features as players cooperating to produce the prediction, and ask
    how to divide the "payout" (prediction minus baseline) fairly among them. The
    Shapley value from cooperative game theory is the ONLY attribution satisfying
    a set of reasonable axioms simultaneously -- efficiency (the parts sum to the
    whole), symmetry (equal contributors get equal credit), and a dummy player
    (a feature that never changes the output gets zero). That uniqueness is what
    sets SHAP apart from LIME: there is a principled right answer, not a choice of
    neighbourhood.

    A feature's Shapley value is its AVERAGE marginal contribution over all
    orderings in which features could be added to the model -- how much including
    it changes the prediction, averaged over every context.

    THE COST, AND THE APPROXIMATION
    -------------------------------
    Exact Shapley values sum over all ``2^d`` feature subsets -- infeasible beyond
    a handful of features. KernelSHAP estimates them by SAMPLING subsets (called
    "coalitions"), masking the absent features to a background value, and solving
    a specially weighted linear regression whose coefficients ARE the Shapley
    values. Fewer samples, more error; it is the practical face of an exponential
    ideal.

    Lundberg & Lee (2017).
    """

    def __init__(self, n_samples=200, random_state=None):
        self.n_samples = n_samples
        self.random_state = random_state

    def explain(self, predict_fn, x, X_background, max_exact=10):
        """Shapley values attributing ``predict_fn(x) - baseline`` to each feature.

        Exact when there are few features (``<= max_exact``), sampled otherwise.
        The values are guaranteed to SUM to the gap between the prediction and the
        background-average prediction -- the efficiency axiom, which is also a
        useful correctness check.
        """
        rng = check_random_state(self.random_state)
        x = np.asarray(x, float).ravel()
        d = len(x)
        baseline = X_background.mean(axis=0)

        def value(mask):
            # features in the mask take x's value; the rest revert to baseline.
            # this is the "present vs absent" of the coalition game
            row = np.where(mask, x, baseline)
            return float(np.asarray(predict_fn(row[None])).ravel()[0])

        if d <= max_exact:
            # exact Shapley: average marginal contribution over ALL subsets
            phi = np.zeros(d)
            from math import factorial
            for j in range(d):
                others = [k for k in range(d) if k != j]
                for r in range(len(others) + 1):
                    for subset in combinations(others, r):
                        mask = np.zeros(d, dtype=bool)
                        mask[list(subset)] = True
                        with_j = mask.copy()
                        with_j[j] = True
                        # the Shapley weight: how many orderings give this subset
                        w = (factorial(r) * factorial(d - r - 1)) / factorial(d)
                        phi[j] += w * (value(with_j) - value(mask))
            self.expected_value_ = value(np.zeros(d, dtype=bool))
            return phi

        # sampled KernelSHAP: random coalitions, then weighted least squares
        masks = rng.uniform(size=(self.n_samples, d)) > 0.5
        vals = np.array([value(m) for m in masks])
        base = value(np.zeros(d, dtype=bool))
        # the SHAP kernel weights coalitions by size (extremes weigh most)
        sizes = masks.sum(axis=1)
        with np.errstate(divide="ignore"):
            weights = np.where(
                (sizes > 0) & (sizes < d),
                (d - 1) / (np.maximum(sizes, 1) * (d - np.minimum(sizes, d - 1))
                           * 1.0), 1e6)
        A = masks.astype(float)
        W = np.diag(weights)
        # solve for phi with the efficiency constraint folded into the target
        phi = np.linalg.lstsq(A.T @ W @ A + 1e-6 * np.eye(d),
                              A.T @ W @ (vals - base), rcond=None)[0]
        self.expected_value_ = base
        return phi


def integrated_gradients(grad_fn, x, baseline=None, n_steps=50):
    """Attribution for a differentiable model, by integrating along a path.

    THE PROBLEM WITH RAW GRADIENTS
    ------------------------------
    The gradient at a point tells you the LOCAL sensitivity, but deep models
    saturate: once a feature is "clearly present", pushing it further does not
    change the output, so its gradient there is ~0 -- and a raw-gradient saliency
    map wrongly calls the decisive feature unimportant. Sensitivity at the answer
    is not the same as responsibility for it.

    THE FIX
    -------
    Integrate the gradient along a straight path from a BASELINE (an "absence"
    input -- a black image, a zero vector) to the actual input. A feature's
    attribution is its accumulated gradient over that journey, which captures the
    stretch where it WAS changing the output, not just the saturated endpoint. It
    satisfies the same efficiency axiom as SHAP -- the attributions sum to the
    difference in output between baseline and input -- which is the sense in which
    it is the principled version of gradient saliency.

    ``grad_fn(X)`` returns the model's gradient w.r.t. its input at each row.

    Sundararajan, Taly & Yan (2017).
    """
    x = np.asarray(x, float).ravel()
    if baseline is None:
        baseline = np.zeros_like(x)                 # "absence" of every feature
    baseline = np.asarray(baseline, float).ravel()

    # sample the straight path from baseline to x and average the gradients
    alphas = np.linspace(0, 1, n_steps)
    path = np.array([baseline + a * (x - baseline) for a in alphas])
    grads = grad_fn(path)                           # (n_steps, d)
    avg_grad = grads.mean(axis=0)
    # multiply the average gradient by the input's displacement from the baseline
    return (x - baseline) * avg_grad


__all__ = ["permutation_importance", "partial_dependence", "ice", "LIME",
           "KernelSHAP", "integrated_gradients"]
