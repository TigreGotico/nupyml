"""Composition: Pipeline, FeatureUnion, ColumnTransformer."""
import numpy as np

from ..base import BaseEstimator, clone, check_is_fitted


class Pipeline(BaseEstimator):
    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final(self):
        return self.steps[-1][1]

    @property
    def _estimator_type(self):
        return getattr(self._final, "_estimator_type", None)

    def get_params(self, deep=True):
        params = {"steps": self.steps}
        if deep:
            for name, est in self.steps:
                if hasattr(est, "get_params"):
                    for k, v in est.get_params(deep=True).items():
                        params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        named = dict(self.steps)
        for key, value in params.items():
            if key == "steps":
                self.steps = value
            elif "__" in key:
                head, _, tail = key.partition("__")
                named[head].set_params(**{tail: value})
            else:
                raise ValueError(f"Invalid parameter {key!r}")
        return self

    def _transform_through(self, X, fit=False, y=None):
        for name, est in self.steps[:-1]:
            if fit:
                X = est.fit_transform(X, y) if hasattr(est, "fit_transform") \
                    else est.fit(X, y).transform(X)
            else:
                X = est.transform(X)
        return X

    def fit(self, X, y=None):
        self.steps = [(name, clone(est)) for name, est in self.steps]
        X = self._transform_through(X, fit=True, y=y)
        self._final.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "fitted_")
        return self._final.predict(self._transform_through(X))

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_")
        return self._final.predict_proba(self._transform_through(X))

    def decision_function(self, X):
        check_is_fitted(self, "fitted_")
        return self._final.decision_function(self._transform_through(X))

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        X = self._transform_through(X)
        return self._final.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def score(self, X, y):
        check_is_fitted(self, "fitted_")
        return self._final.score(self._transform_through(X), y)

    def __getitem__(self, name):
        return self.named_steps[name]


def make_pipeline(*steps):
    return Pipeline([(type(s).__name__.lower(), s) for s in steps])


class FeatureUnion(BaseEstimator):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        self.transformer_list = [(n, clone(t).fit(X, y))
                                 for n, t in self.transformer_list]
        return self

    def transform(self, X):
        return np.hstack([np.asarray(t.transform(X))
                          for _, t in self.transformer_list])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ColumnTransformer(BaseEstimator):
    """Apply transformers to column subsets: [(name, transformer, columns)]."""

    def __init__(self, transformers, remainder="drop"):
        self.transformers = transformers
        self.remainder = remainder

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.transformers_ = []
        used = set()
        for name, t, cols in self.transformers:
            cols = list(cols)
            used.update(cols)
            self.transformers_.append((name, clone(t).fit(X[:, cols], y), cols))
        self._passthrough = [j for j in range(X.shape[1]) if j not in used] \
            if self.remainder == "passthrough" else []
        return self

    def transform(self, X):
        check_is_fitted(self, "transformers_")
        X = np.asarray(X)
        parts = [np.asarray(t.transform(X[:, cols]))
                 for _, t, cols in self.transformers_]
        if self._passthrough:
            parts.append(X[:, self._passthrough].astype(np.float64))
        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


__all__ = ["Pipeline", "make_pipeline", "FeatureUnion", "ColumnTransformer"]
