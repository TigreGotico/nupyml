"""Local, example-centred explanations: surrogate trees, anchors, counterfactuals.

Where ``attribution`` attributes a prediction to features NUMERICALLY (SHAP,
LIME), these give RULE- and EXAMPLE-shaped answers: a global tree that mimics the
box, an IF-THEN rule that pins a prediction down, and the nearest input that would
have flipped it.
"""
import numpy as np

from ..utils import check_array, check_random_state


def _predict(model, X):
    return np.asarray(model.predict(X))


def surrogate_tree(model, X, max_depth=3, task="classification", random_state=None):
    """Fit an interpretable decision tree to the black box's OWN predictions.

    A global surrogate approximates any model with something you can read: label
    the data with the BLACK BOX's predictions (not the true labels), fit a shallow
    tree to those, and read the tree as a summary of the model's logic. The
    ``fidelity_`` it returns -- how often the tree agrees with the box -- tells you
    how much to trust the summary; a low fidelity means the box is too complex to
    be captured by a small tree, which is itself useful to know.

    Returns (tree, fidelity).
    """
    from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
    X = check_array(X)
    y_box = _predict(model, X)
    if task == "classification":
        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=random_state).fit(X, y_box)
        fidelity = float(np.mean(tree.predict(X) == y_box))
    else:
        tree = DecisionTreeRegressor(max_depth=max_depth,
                                     random_state=random_state).fit(X, y_box)
        ss_res = np.sum((y_box - tree.predict(X)) ** 2)
        ss_tot = np.sum((y_box - y_box.mean()) ** 2) + 1e-12
        fidelity = float(1 - ss_res / ss_tot)          # R^2 fidelity
    return tree, fidelity


class Anchors:
    """Find a high-precision IF-THEN rule that ANCHORS a prediction.

    THE IDEA
    --------
    An anchor for an instance is a small set of conditions (``feature in
    [lo, hi]``) such that ANY point satisfying them gets the same prediction, with
    high probability. "IF age>50 AND status=married THEN approve, 97% of the time"
    -- a rule a human can check, unlike a bag of SHAP weights. Precision is how
    reliably the rule implies the label; coverage is how much of the space it
    applies to.

    HOW
    ---
    Greedily grow the rule: start from no conditions and repeatedly add the
    predicate (this instance's value of some feature, widened to a quantile bin)
    that best keeps precision high on random perturbations, until precision clears
    ``threshold``. Greedy anchoring is the practical core of the Ribeiro et al.
    (2018) method.
    """

    def __init__(self, model, threshold=0.95, n_samples=500, random_state=None):
        self.model = model
        self.threshold = threshold
        self.n_samples = n_samples
        self.random_state = random_state

    def explain(self, x, X_background):
        x = np.asarray(x, float)
        Xb = check_array(X_background)
        rng = check_random_state(self.random_state)
        target = _predict(self.model, x.reshape(1, -1))[0]
        d = len(x)
        # candidate predicate per feature: a quantile band around x_j
        bands = {}
        for j in range(d):
            lo = np.percentile(Xb[:, j], 25)
            hi = np.percentile(Xb[:, j], 75)
            width = (hi - lo) or 1.0
            bands[j] = (x[j] - width / 2, x[j] + width / 2)

        chosen = {}
        remaining = set(range(d))
        for _ in range(d):
            best_j, best_prec = None, -1.0
            for j in remaining:
                rule = dict(chosen); rule[j] = bands[j]
                prec = self._precision(rule, Xb, rng, target)
                if prec > best_prec:
                    best_prec, best_j = prec, j
            chosen[best_j] = bands[best_j]
            remaining.discard(best_j)
            if self._precision(chosen, Xb, rng, target) >= self.threshold:
                break
        self.rule_ = chosen
        self.precision_ = self._precision(chosen, Xb, rng, target)
        self.coverage_ = np.mean([self._matches(chosen, row) for row in Xb])
        return self

    def _matches(self, rule, row):
        return all(lo <= row[j] <= hi for j, (lo, hi) in rule.items())

    def _precision(self, rule, Xb, rng, target):
        # sample background rows that satisfy the rule; how often does the model
        # still predict the target?
        idx = [i for i, row in enumerate(Xb) if self._matches(rule, row)]
        if not idx:
            return 0.0
        take = rng.choice(idx, size=min(self.n_samples, len(idx)), replace=len(idx) < 5)
        preds = _predict(self.model, Xb[take])
        return float(np.mean(preds == target))


def counterfactual(model, x, X_background, desired_class=None, n_iter=2000,
                   random_state=None):
    """Find the NEAREST input that flips the prediction -- an actionable "what-if".

    A counterfactual answers "what is the smallest change to this instance that
    would change the decision?" -- the recourse an affected person actually wants
    ("earn $5k more and the loan is approved"). This does a growing-search: sample
    perturbations of increasing radius (scaled per feature by the background
    spread) and keep the closest one that reaches the desired class.

    Wachter et al. (2017). Returns (x_cf, distance) or (None, inf) if none found.
    """
    x = np.asarray(x, float)
    Xb = check_array(X_background)
    rng = check_random_state(random_state)
    scale = Xb.std(axis=0) + 1e-9
    current = _predict(model, x.reshape(1, -1))[0]
    if desired_class is None:
        classes = getattr(model, "classes_", np.unique(_predict(model, Xb)))
        desired_class = next(c for c in classes if c != current)

    best, best_d = None, np.inf
    for t in range(1, n_iter + 1):
        radius = scale * (t / n_iter) * 3.0            # grow the search sphere
        cand = x + rng.randn(len(x)) * radius
        if _predict(model, cand.reshape(1, -1))[0] == desired_class:
            d = np.sqrt(np.sum(((cand - x) / scale) ** 2))
            if d < best_d:
                best_d, best = d, cand
    return best, best_d


__all__ = ["surrogate_tree", "Anchors", "counterfactual"]
