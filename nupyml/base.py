"""The estimator contract that everything in this library obeys.

One set of conventions is what lets a ``Pipeline`` hold any transformer, a
``GridSearchCV`` tune any model, and ``clone`` copy any estimator without
knowing what it is:

* **Parameters go in ``__init__``, and are stored unchanged.** No validation, no
  coercion, no computation. ``clone`` reconstructs an estimator by reading
  ``get_params`` and calling ``__init__`` with them, so anything ``__init__``
  alters is silently re-altered on every clone -- which is why the rule is
  strict, and why ``check_estimator`` tests it.
* **Learning happens in ``fit``, and only in ``fit``.**
* **Learned attributes end in an underscore** (``coef_``, ``labels_``). That
  trailing underscore is how ``check_is_fitted`` can tell a fitted estimator
  from a fresh one without either being told.
* **``fit`` returns self**, so ``Model().fit(X, y).predict(X)`` chains.

The mixins supply behaviour that follows from those conventions:
``ClassifierMixin`` and ``RegressorMixin`` give a default ``score``,
``TransformerMixin`` gives ``fit_transform``, and ``_SetOutputMixin`` provides
the optional pandas integration.

``utils.estimator_checks.check_estimator`` verifies all of it, and is worth
reading as an executable statement of the contract.
"""
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

    def _repr_html_(self):
        """Rendered by Jupyter: the class, its parameters, and fitted state."""
        import html as _html
        name = type(self).__name__
        fitted = any(k.endswith("_") and not k.startswith("_")
                     for k in vars(self))
        rows = "".join(
            f"<tr><td style='padding:2px 8px;color:#555'>{_html.escape(k)}</td>"
            f"<td style='padding:2px 8px'><code>{_html.escape(repr(v))}</code>"
            f"</td></tr>"
            for k, v in sorted(self.get_params(deep=False).items()))
        badge = ("<span style='color:#2b8a3e'>fitted</span>" if fitted
                 else "<span style='color:#999'>not fitted</span>")
        doc = (type(self).__doc__ or "").strip().split("\n")[0]
        return (
            f"<div style='border:1px solid #ddd;border-radius:4px;"
            f"padding:8px;display:inline-block;font-family:sans-serif'>"
            f"<div style='font-weight:600'>{name} &nbsp;{badge}</div>"
            f"<div style='color:#666;font-size:90%;margin:2px 0 6px'>"
            f"{_html.escape(doc)}</div>"
            f"<table style='font-size:90%'>{rows}</table></div>")

    def _check_feature_names(self, X, reset=False):
        """Remember the column names of a dataframe and flag mismatches."""
        names = getattr(X, "columns", None)
        names = None if names is None else np.asarray([str(c) for c in names])
        if reset:
            if names is not None:
                self.feature_names_in_ = names
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
            return
        seen = getattr(self, "feature_names_in_", None)
        if seen is not None and names is not None and not np.array_equal(seen,
                                                                         names):
            raise ValueError(
                f"The feature names should match those seen during fit.\n"
                f"Fitted with: {list(seen)}\nGot: {list(names)}")

    def get_feature_names_out(self, input_features=None):
        """Default: a transformer keeps one output column per input column."""
        n_out = getattr(self, "n_features_out_", None)
        if input_features is not None:
            input_features = np.asarray([str(c) for c in input_features])
        elif hasattr(self, "feature_names_in_"):
            input_features = self.feature_names_in_
        if n_out is None and input_features is not None:
            return input_features
        n_in = getattr(self, "n_features_in_", None)
        if n_out is None:
            n_out = n_in
        if n_out is None:
            raise RuntimeError(
                f"{type(self).__name__} cannot report feature names before fit")
        if input_features is not None and len(input_features) == n_out:
            return input_features
        prefix = type(self).__name__.lower()
        return np.array([f"{prefix}{i}" for i in range(n_out)])


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


class _SetOutputMixin:
    """Opt-in pandas output. pandas stays optional: without it, or without
    set_output, everything remains plain numpy."""

    def set_output(self, transform=None):
        if transform not in (None, "default", "pandas"):
            raise ValueError(
                f"transform must be 'default' or 'pandas', got {transform!r}")
        if transform is not None:
            self._output_config = transform
        return self

    def _wrap_output(self, X_out, X_in):
        if getattr(self, "_output_config", "default") != "pandas":
            return X_out
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "set_output(transform='pandas') needs pandas installed") from exc
        if isinstance(X_out, pd.DataFrame):
            return X_out
        arr = np.asarray(X_out)
        try:
            columns = self.get_feature_names_out()
        except Exception:
            columns = None
        if columns is not None and len(columns) != arr.shape[1]:
            columns = None
        index = getattr(X_in, "index", None)
        return pd.DataFrame(arr, columns=columns, index=index)


class TransformerMixin(_SetOutputMixin):
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def __init_subclass__(cls, **kwargs):
        """Wrap fit/transform once per subclass so every transformer records
        dataframe column names and honours set_output without each one having
        to remember to."""
        super().__init_subclass__(**kwargs)
        for meth in ("fit", "transform"):
            fn = cls.__dict__.get(meth)
            if fn is None or getattr(fn, "_nupyml_wrapped", False):
                continue
            setattr(cls, meth, _wrap_io(fn, meth))


def _wrap_io(fn, kind):
    """Add feature-name tracking and set_output around fit/transform without
    assuming the wrapped signature: some transformers take no X at all."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        X = args[0] if args else kwargs.get("X")
        if isinstance(self, BaseEstimator) and hasattr(X, "columns"):
            self._check_feature_names(X, reset=(kind == "fit"))
        out = fn(self, *args, **kwargs)
        if kind == "transform" and isinstance(self, _SetOutputMixin):
            return self._wrap_output(out, X)
        return out

    wrapper._nupyml_wrapped = True
    return wrapper


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
