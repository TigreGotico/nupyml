"""A conformance suite for estimators: does this class honour the API contract?"""
import inspect
import warnings

import numpy as np

from ..base import clone


def _tags(estimator):
    """Estimator tags describe what data an estimator can legally accept, so
    the harness can build valid input instead of tripping over its own data."""
    default = {"binary_only": False, "requires_positive_y": False,
               "requires_positive_X": False, "non_deterministic": False}
    default.update(getattr(estimator, "_estimator_tags", {}))
    return default


def _make_data(estimator):
    from ..datasets import make_blobs, make_regression
    tags = _tags(estimator)
    kind = getattr(estimator, "_estimator_type", None)
    if kind == "regressor":
        X, y = make_regression(n_samples=80, n_features=4, noise=1.0,
                               random_state=0)
        y = y / np.abs(y).max()
        if tags["requires_positive_y"]:
            y = np.abs(y) + 0.5
        return (np.abs(X) if tags["requires_positive_X"] else X), y
    n_centers = 2 if tags["binary_only"] else 3
    X, y = make_blobs(n_samples=90, n_features=4, centers=n_centers,
                      cluster_std=1.0, random_state=0)
    if tags["requires_positive_X"]:
        X = np.abs(X)
    return X, y


def _is_unsupervised(estimator):
    """Unsupervised fits either take no y at all, or default it to None."""
    params = list(inspect.signature(estimator.fit).parameters.values())
    if len(params) < 2:
        return True
    second = params[1]
    return second.name == "y" and second.default is not inspect.Parameter.empty


def _fit(estimator, X, y):
    if _is_unsupervised(estimator):
        return estimator.fit(X)
    return estimator.fit(X, y)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_get_params_returns_init_args(name, estimator):
    """get_params must expose exactly the constructor arguments."""
    init_params = set(type(estimator)._get_param_names())
    got = set(estimator.get_params(deep=False))
    if init_params != got:
        raise AssertionError(
            f"{name}: get_params returned {sorted(got)}, expected "
            f"{sorted(init_params)}")


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    # nan != nan, but a nan default stored as nan is unchanged
    if isinstance(a, float) and isinstance(b, float) \
            and np.isnan(a) and np.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return a is b


def check_init_does_not_modify_params(name, estimator):
    """__init__ must store parameters unchanged: no validation, no coercion.

    Compares against the signature defaults rather than get_params, since
    get_params reads back the very attribute a buggy __init__ mutated.
    """
    cls = type(estimator)
    defaults = {
        p.name: p.default
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self" and p.default is not inspect.Parameter.empty
    }
    fresh = cls()
    for key, default in defaults.items():
        stored = getattr(fresh, key, "<missing>")
        if not _equal(stored, default):
            raise AssertionError(
                f"{name}: __init__ changed parameter {key!r} from its default "
                f"{default!r} to {stored!r}; validation belongs in fit")
    # and explicitly-passed values must survive untouched too
    for key, default in defaults.items():
        probe = _probe_value(default)
        if probe is None:
            continue
        try:
            est = cls(**{key: probe})
        except (TypeError, ValueError):
            continue
        if not _equal(getattr(est, key, "<missing>"), probe):
            raise AssertionError(
                f"{name}: __init__ did not store {key}={probe!r} unchanged")


def _probe_value(default):
    """A distinct-but-plausible value of the same type as ``default``."""
    if isinstance(default, bool) or default is None:
        return None
    if isinstance(default, int):
        return default + 3
    if isinstance(default, float):
        return default + 0.5 if default else 0.5
    return None


def check_set_params_roundtrip(name, estimator):
    params = estimator.get_params(deep=False)
    est = clone(estimator).set_params(**params)
    for key, value in params.items():
        got = getattr(est, key)
        if isinstance(value, np.ndarray):
            continue
        if got is not value and got != value:
            raise AssertionError(f"{name}: set_params lost {key!r}")


def check_clone_is_independent(name, estimator):
    cloned = clone(estimator)
    if cloned is estimator:
        raise AssertionError(f"{name}: clone returned the same object")
    if type(cloned) is not type(estimator):
        raise AssertionError(f"{name}: clone changed the type")
    if any(k.endswith("_") and not k.startswith("__") for k in vars(cloned)):
        raise AssertionError(f"{name}: clone carried over fitted attributes")


def check_fit_returns_self(name, estimator):
    X, y = _make_data(estimator)
    est = clone(estimator)
    out = _fit(est, X, y)
    if out is not est:
        raise AssertionError(f"{name}: fit must return self")


def check_fitted_attributes_have_trailing_underscore(name, estimator):
    X, y = _make_data(estimator)
    before = set(vars(clone(estimator)))
    est = clone(estimator)
    _fit(est, X, y)
    new = set(vars(est)) - before
    public_new = {k for k in new if not k.startswith("_")}
    if public_new and not any(k.endswith("_") for k in public_new):
        raise AssertionError(
            f"{name}: fit set public attributes without a trailing "
            f"underscore: {sorted(public_new)}")


def check_predict_before_fit_raises(name, estimator):
    if not hasattr(estimator, "predict"):
        return
    X, _ = _make_data(estimator)
    est = clone(estimator)
    try:
        est.predict(X)
    except (RuntimeError, AttributeError, ValueError):
        return
    raise AssertionError(f"{name}: predict before fit must raise")


def check_predict_shape_and_dtype(name, estimator):
    if not hasattr(estimator, "predict"):
        return
    X, y = _make_data(estimator)
    est = clone(estimator)
    _fit(est, X, y)
    pred = est.predict(X)
    if len(pred) != len(X):
        raise AssertionError(
            f"{name}: predict returned {len(pred)} rows for {len(X)} samples")
    if getattr(est, "_estimator_type", None) == "classifier":
        unseen = set(np.unique(pred)) - set(np.unique(est.classes_))
        if unseen:
            raise AssertionError(f"{name}: predict emitted unknown labels {unseen}")


def check_predict_proba_is_a_distribution(name, estimator):
    if not hasattr(estimator, "predict_proba"):
        return
    X, y = _make_data(estimator)
    est = clone(estimator)
    _fit(est, X, y)
    try:
        proba = est.predict_proba(X)
    except AttributeError:
        return          # legitimately gated (e.g. SVC(probability=False))
    proba = np.asarray(proba)
    if proba.ndim != 2:
        return          # multi-output estimators return a list
    if not np.allclose(proba.sum(axis=1), 1):
        raise AssertionError(f"{name}: predict_proba rows must sum to 1")
    if (proba < -1e-9).any() or (proba > 1 + 1e-9).any():
        raise AssertionError(f"{name}: predict_proba outside [0, 1]")


def check_fit_is_deterministic(name, estimator):
    if "random_state" not in type(estimator)._get_param_names():
        return
    X, y = _make_data(estimator)
    a = clone(estimator).set_params(random_state=0)
    b = clone(estimator).set_params(random_state=0)
    _fit(a, X, y)
    _fit(b, X, y)
    if hasattr(a, "predict"):
        if not np.array_equal(np.asarray(a.predict(X)), np.asarray(b.predict(X))):
            raise AssertionError(
                f"{name}: same random_state produced different predictions")


def check_refit_overwrites(name, estimator):
    """A second fit must not blend with the first.

    Only meaningful once randomness is pinned: with random_state=None a refit
    is legitimately allowed to land somewhere else.
    """
    if _tags(estimator)["non_deterministic"]:
        return
    base = clone(estimator)
    if "random_state" in type(estimator)._get_param_names():
        base.set_params(random_state=0)
    if getattr(base, "warm_start", False):
        return
    X, y = _make_data(base)
    est = clone(base)
    _fit(est, X, y)
    _fit(est, X, y)
    fresh = clone(base)
    _fit(fresh, X, y)
    if hasattr(est, "predict"):
        if not np.array_equal(np.asarray(est.predict(X)),
                              np.asarray(fresh.predict(X))):
            raise AssertionError(
                f"{name}: refitting did not reset the model state")


def check_transform_shape(name, estimator):
    if not hasattr(estimator, "transform"):
        return
    X, y = _make_data(estimator)
    est = clone(estimator)
    _fit(est, X, y)
    Xt = np.asarray(est.transform(X))
    if len(Xt) != len(X):
        raise AssertionError(f"{name}: transform changed the number of rows")


def check_repr_does_not_raise(name, estimator):
    text = repr(estimator)
    if type(estimator).__name__ not in text:
        raise AssertionError(f"{name}: repr should name the class")


_CHECKS = [
    check_get_params_returns_init_args,
    check_init_does_not_modify_params,
    check_set_params_roundtrip,
    check_clone_is_independent,
    check_repr_does_not_raise,
    check_fit_returns_self,
    check_fitted_attributes_have_trailing_underscore,
    check_predict_before_fit_raises,
    check_predict_shape_and_dtype,
    check_predict_proba_is_a_distribution,
    check_transform_shape,
    check_fit_is_deterministic,
    check_refit_overwrites,
]


def check_estimator(estimator, generate_only=False):
    """Run every conformance check against ``estimator``.

    Returns the list of check names that ran; raises AssertionError on the
    first violation. With ``generate_only=True`` yields (estimator, check)
    pairs instead of running them.
    """
    name = type(estimator).__name__
    if generate_only:
        return [(estimator, check) for check in _CHECKS]
    passed = []
    for check in _CHECKS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check(name, estimator)
        passed.append(check.__name__)
    return passed


__all__ = ["check_estimator", "_tags"]
