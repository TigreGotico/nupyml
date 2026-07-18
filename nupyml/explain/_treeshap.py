"""Exact SHAP values for a decision tree, read off the tree structure.

KernelSHAP (in ``attribution``) estimates Shapley values by sampling coalitions
-- slow and approximate. For a TREE the exact Shapley values follow from the tree
itself: the expected output with any subset of features fixed is one recursive
pass (follow fixed features; average the rest by each node's training coverage,
which the tree already stores). This is the transparent form of TreeSHAP.
"""
import numpy as np
from itertools import combinations
from math import factorial

from ..utils import check_array


class TreeSHAP:
    """Exact Shapley feature attributions for a single nupyml decision tree.

    THE SHORTCUT
    ------------
    A Shapley value averages a feature's marginal contribution over ALL feature
    orderings -- exponentially many in general. For a tree the model IS the tree:
    the expected prediction with a subset ``S`` of features fixed to the query's
    values is computed by descending the tree, following the split when its
    feature is in ``S`` and otherwise averaging both children weighted by their
    training COVERAGE (stored on each node). TreeSHAP uses that to get EXACT
    Shapley values. This version sums the coalition contributions directly over
    the features the tree actually uses -- exact, and the readable version of the
    linear-time algorithm.

    ``explain(x)`` returns one SHAP value per feature; by the efficiency axiom they
    SUM to ``prediction(x) - base_value_`` (checked in the tests).
    """

    def __init__(self, tree_model):
        node = getattr(tree_model, "tree_", None)
        if node is None:
            raise ValueError("TreeSHAP needs a fitted nupyml decision tree")
        self._root = self._convert(node)
        self._feats = sorted(self._features(self._root))
        # base value: everything marginalised out (no features fixed)
        self.base_value_ = self._expected(self._root, None, frozenset())

    def _leaf_value(self, node):
        v = np.ravel(node.value)
        return float(v[0]) if v.size == 1 else float(np.argmax(v))

    def _convert(self, node):
        if node.is_leaf:
            return {"leaf": True, "value": self._leaf_value(node)}
        nl = node.left.n_samples
        nr = node.right.n_samples
        total = nl + nr
        return {"leaf": False, "feature": node.feature, "threshold": node.threshold,
                "w_left": nl / total, "w_right": nr / total,
                "left": self._convert(node.left), "right": self._convert(node.right)}

    def _expected(self, node, x, S):
        if node["leaf"]:
            return node["value"]
        f = node["feature"]
        if x is not None and f in S:                  # feature fixed -> follow it
            child = node["left"] if x[f] <= node["threshold"] else node["right"]
            return self._expected(child, x, S)
        # feature absent -> marginalise over children by training coverage
        return (node["w_left"] * self._expected(node["left"], x, S)
                + node["w_right"] * self._expected(node["right"], x, S))

    def _features(self, node, acc=None):
        acc = set() if acc is None else acc
        if not node["leaf"]:
            acc.add(node["feature"])
            self._features(node["left"], acc)
            self._features(node["right"], acc)
        return acc

    def explain(self, x):
        x = np.asarray(x, float)
        n_features_out = int(max(self._feats)) + 1 if self._feats else 1
        shap = np.zeros(max(n_features_out, len(x)))
        m = len(self._feats)
        for f in self._feats:
            others = [g for g in self._feats if g != f]
            contrib = 0.0
            for r in range(len(others) + 1):
                weight = factorial(r) * factorial(m - r - 1) / factorial(m)
                for subset in combinations(others, r):
                    S = frozenset(subset)
                    contrib += weight * (self._expected(self._root, x, S | {f})
                                         - self._expected(self._root, x, S))
            shap[f] = contrib
        return shap[:len(x)]

    def predict(self, x):
        """The tree's own prediction (all features fixed) -- base + sum(shap)."""
        return self._expected(self._root, np.asarray(x, float),
                              frozenset(self._feats))


__all__ = ["TreeSHAP"]
