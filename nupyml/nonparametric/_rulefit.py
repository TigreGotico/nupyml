"""RuleFit: turn a forest into a short list of rules a person can read.

THE IDEA
--------
A tree is accurate but opaque; a rule -- "IF age > 40 AND income < 30k THEN ..."
-- is transparent but weak alone. RuleFit gets both:

1. Fit a tree ENSEMBLE (accurate, opaque).
2. Read every root-to-node path out of it as a binary rule. Each rule is a
   feature: 1 where it fires, 0 otherwise.
3. Fit a LASSO on those rule-features (plus the raw features, linearly).

The lasso does the real work. A forest of a few hundred trees yields thousands of
rules, almost all redundant or useless. L1 regularisation drives nearly all their
coefficients to exactly zero and keeps a handful. The result is a linear model
over a dozen interpretable rules: nearly the forest's accuracy, with an
explanation you can print.

WHY BOTH RULES AND LINEAR TERMS
-------------------------------
Rules capture interactions and sharp thresholds -- "young AND urban", "above the
limit". Linear terms capture smooth global trends, which rules approximate only
with an ugly staircase of thresholds. Including both lets each describe what it
describes well, and lets the lasso choose per effect. A purely rule-based model
wastes coefficients staircasing a straight line.

WHY THE LASSO AND NOT THE RIDGE
-------------------------------
The whole point is a SHORT list. L2 would shrink every rule a little and keep all
thousands -- accurate, and no more interpretable than the forest it came from.
L1's corner solutions, setting coefficients to exactly zero, are what make the
result readable. This is the clearest payoff of the L1-vs-L2 distinction from
``linear_model``: here sparsity is not a nicety, it is the entire product.

Friedman & Popescu (2008).
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, ClassifierMixin, check_is_fitted
from ..ensemble import GradientBoostingRegressor
from ..linear_model import Lasso, LogisticRegression
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..utils import check_X_y, check_array


class Rule:
    """A conjunction of thresholds read from one path down a tree.

    ``conditions`` is a list of ``(feature, op, threshold)`` with ``op`` in
    ``{"<=", ">"}``. The rule fires (outputs 1) when every condition holds.
    """

    def __init__(self, conditions):
        self.conditions = conditions

    def evaluate(self, X):
        out = np.ones(len(X), dtype=np.float64)
        for feature, op, thresh in self.conditions:
            if op == "<=":
                out *= (X[:, feature] <= thresh)
            else:
                out *= (X[:, feature] > thresh)
        return out

    def __str__(self):
        return " AND ".join(f"x{f} {op} {t:.3g}" for f, op, t in self.conditions)

    def __len__(self):
        return len(self.conditions)


def _extract_rules(tree, max_depth=None):
    """Every root-to-internal-node path as a Rule.

    Each edge adds one condition. Interior nodes (not just leaves) are emitted
    too: a short rule from a shallow node is often more useful and more general
    than the long one reaching a leaf, and the lasso should get to choose between
    them rather than have the depth decided here.
    """
    rules = []

    def walk(node, conditions):
        if node.is_leaf:
            return
        if max_depth is not None and len(conditions) >= max_depth:
            return
        feat, thresh = node.feature, node.threshold
        left = conditions + [(feat, "<=", thresh)]
        right = conditions + [(feat, ">", thresh)]
        if conditions:                      # skip the empty root "rule"
            rules.append(Rule(list(conditions)))
        walk(node.left, left)
        walk(node.right, right)

    walk(tree.tree_, [])
    # the deepest paths are never emitted by the loop above (they end at leaves),
    # so add each leaf's own path
    def walk_leaves(node, conditions):
        if node.is_leaf:
            if conditions:
                rules.append(Rule(list(conditions)))
            return
        walk_leaves(node.left, conditions + [(node.feature, "<=", node.threshold)])
        walk_leaves(node.right, conditions + [(node.feature, ">", node.threshold)])

    walk_leaves(tree.tree_, [])
    return rules


class RuleFit(BaseEstimator, RegressorMixin):
    """Rule ensemble + lasso: a forest's accuracy, a short list as the reason.

    After ``fit``, ``rules_`` holds the surviving rules with non-zero weight,
    sorted by importance. Printing them is the point of the method.
    """

    def __init__(self, n_estimators=100, max_depth=3, alpha=0.1,
                 include_linear=True, max_rules=2000, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.include_linear = include_linear
        self.max_rules = max_rules
        self.random_state = random_state

    def _build_rules(self, X, y):
        """Grow a tree ensemble and read all its paths out as rules."""
        gb = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=0.1, random_state=self.random_state).fit(X, y)
        rules = []
        for est in gb.estimators_:
            tree = est[0] if isinstance(est, (list, tuple, np.ndarray)) else est
            rules.extend(_extract_rules(tree, self.max_depth))
        return rules

    def _design(self, X, rules):
        cols = [r.evaluate(X) for r in rules]
        if self.include_linear:
            # standardise linear terms so the single alpha penalises rules and
            # linear effects on a comparable scale -- rules are 0/1, raw features
            # are whatever their units are, and the lasso would otherwise punish
            # a feature for being measured in small numbers
            linear = (X - self._x_mean) / self._x_std
            cols = [linear[:, j] for j in range(X.shape[1])] + cols
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0

        rules = self._build_rules(X, y)
        # deduplicate: a forest produces the same rule many times, and identical
        # columns give the lasso nothing to choose between while slowing it down
        seen, unique = set(), []
        for r in rules:
            key = tuple(sorted(r.conditions))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        self._all_rules = unique[:self.max_rules]

        design = self._design(X, self._all_rules)
        self._lasso = Lasso(alpha=self.alpha).fit(design, y)

        # the sparsity is the product: keep only what survived
        coef = self._lasso.coef_
        n_lin = self.n_features_in_ if self.include_linear else 0
        rule_coef = coef[n_lin:]
        keep = np.where(np.abs(rule_coef) > 1e-10)[0]
        order = keep[np.argsort(-np.abs(rule_coef[keep]))]
        self.rules_ = [self._all_rules[i] for i in order]
        self.rule_weights_ = rule_coef[order]
        self.linear_coef_ = coef[:n_lin]
        self.intercept_ = self._lasso.intercept_
        self.n_rules_ = len(self.rules_)
        return self

    def predict(self, X):
        check_is_fitted(self, "rules_")
        X = check_array(X)
        design = self._design(X, self._all_rules)
        return self._lasso.predict(design)

    def summary(self, top=10):
        """The surviving rules and their weights -- the model, made readable."""
        check_is_fitted(self, "rules_")
        lines = [f"intercept: {self.intercept_:.4g}"]
        for r, w in list(zip(self.rules_, self.rule_weights_))[:top]:
            lines.append(f"{w:+.4g}  IF {r}")
        return "\n".join(lines)


__all__ = ["RuleFit", "Rule"]
