"""Deep feature synthesis: automated feature engineering by aggregation.

Much predictive signal lives in AGGREGATES over related rows -- a customer's
average order value, their number of orders, the std of their session lengths.
Building these by hand is tedious and easy to get wrong. Deep feature synthesis
(the idea behind featuretools) generates them mechanically from an entity's
related rows.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array


_PRIMITIVES = {
    "mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min,
    "std": np.std, "count": len, "median": np.median,
}


class DeepFeatureSynthesis(BaseEstimator, TransformerMixin):
    """Generate per-group AGGREGATION features (featuretools-lite).

    THE IDEA
    --------
    Given a table where each row belongs to an ENTITY (a customer, a device) via a
    group key, and numeric value columns, deep feature synthesis applies
    AGGREGATION PRIMITIVES (mean, sum, max, min, std, count) to each entity's rows
    and attaches the results back to every row of that entity. So a raw
    transactions table yields, per row, "this customer's mean spend", "their max
    spend", "their transaction count", and so on -- the relational features that
    usually carry the signal, generated automatically instead of by hand. Learned
    aggregates from training are stored, so test rows (including unseen entities,
    which fall back to the global aggregate) transform consistently.

    ``fit(X)`` where column ``group_col`` is the entity id and the rest are values;
    ``transform`` returns the original values plus the aggregation features.
    """

    def __init__(self, group_col=0, primitives=("mean", "std", "max", "min", "count")):
        self.group_col = group_col
        self.primitives = primitives

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        groups = X[:, self.group_col]
        value_cols = [c for c in range(X.shape[1]) if c != self.group_col]
        self.value_cols_ = value_cols
        self.tables_ = {}                             # group -> aggregate vector
        self.global_ = self._agg(X[:, value_cols], X[:, value_cols])
        for g in np.unique(groups):
            rows = X[groups == g][:, value_cols]
            self.tables_[g] = self._agg(rows, X[:, value_cols])
        self.n_out_ = len(self.value_cols_) + len(self.value_cols_) * (
            len(self.primitives) - (1 if "count" in self.primitives else 0)) + (
            1 if "count" in self.primitives else 0)
        return self

    def _agg(self, rows, all_rows):
        feats = []
        for prim in self.primitives:
            fn = _PRIMITIVES[prim]
            if prim == "count":
                feats.append(float(len(rows)))
            else:
                for c in range(rows.shape[1]):
                    feats.append(float(fn(rows[:, c])) if len(rows) else
                                 float(fn(all_rows[:, c])))
        return np.array(feats)

    def transform(self, X):
        check_is_fitted(self, "tables_")
        X = np.asarray(X, dtype=float)
        groups = X[:, self.group_col]
        base = X[:, self.value_cols_]
        agg = np.array([self.tables_.get(g, self.global_) for g in groups])
        return np.hstack([base, agg])


__all__ = ["DeepFeatureSynthesis"]
