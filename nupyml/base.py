"""Base classes and estimator utilities (sklearn-compatible conventions)."""
import copy
import inspect

import numpy as np


class BaseEstimator:
    """Base class for all estimators.

    Follows the scikit-learn convention: all constructor parameters are
    stored as attributes with the same name and never modified in ``fit``.
    """

    @classmethod
    def _get_param_names(cls):
        init = cls.__init__
        if init is object.__init__:
            return []
        sig = inspect.signature(init)
        return sorted(
            p.name for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )

    def get_params(self, deep=True):
        params = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            params[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for k, v in value.get_params(deep=True).items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        if not params:
            return self
        valid = set(self._get_param_names())
        nested = {}
        for key, value in params.items():
            if "__" in key:
                head, _, tail = key.partition("__")
                nested.setdefault(head, {})[tail] = value
            elif key in valid:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}"
                )
        for head, sub in nested.items():
            if head not in valid:
                raise ValueError(
                    f"Invalid parameter {head!r} for estimator {type(self).__name__}"
                )
            getattr(self, head).set_params(**sub)
        return self

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params(deep=False).items()))
        return f"{type(self).__name__}({params})"


def clone(estimator):
    """Return an unfitted copy of ``estimator`` with the same parameters."""
    if isinstance(estimator, (list, tuple)):
        return type(estimator)(clone(e) for e in estimator)
    if not hasattr(estimator, "get_params"):
        return copy.deepcopy(estimator)
    params = estimator.get_params(deep=False)
    return type(estimator)(**{k: clone(v) if hasattr(v, "get_params") else copy.deepcopy(v)
                              for k, v in params.items()})


def is_fitted(estimator):
    return any(k.endswith("_") and not k.startswith("_") for k in vars(estimator))


def check_is_fitted(estimator, attributes=None):
    if attributes is not None:
        if isinstance(attributes, str):
            attributes = [attributes]
        fitted = all(hasattr(estimator, a) for a in attributes)
    else:
        fitted = is_fitted(estimator)
    if not fitted:
        raise RuntimeError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )


class ClassifierMixin:
    _estimator_type = "classifier"

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class RegressorMixin:
    _estimator_type = "regressor"

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class ClusterMixin:
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


class DensityMixin:
    _estimator_type = "density_estimator"


__all__ = [
    "BaseEstimator", "ClassifierMixin", "RegressorMixin", "TransformerMixin",
    "ClusterMixin", "DensityMixin", "clone", "check_is_fitted", "is_fitted",
]
