"""Validation and random-state utilities."""
import numbers

import numpy as np
import scipy.sparse as sp


def check_random_state(seed):
    """Turn seed into a ``np.random.RandomState`` instance."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a RandomState instance")


def check_array(X, dtype=np.float64, accept_sparse=False, ensure_2d=True,
                allow_nd=False, copy=False):
    """Input validation on an array-like."""
    if sp.issparse(X):
        if not accept_sparse:
            raise TypeError(
                "Sparse input is not supported by this estimator; "
                "convert with .toarray()"
            )
        X = X.tocsr().astype(dtype, copy=copy) if dtype is not None else X.tocsr()
        return X
    X = np.array(X, dtype=dtype) if copy else np.asarray(X, dtype=dtype)
    if ensure_2d:
        if X.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got 1D array instead: {X!r}. "
                "Reshape your data with .reshape(-1, 1) or .reshape(1, -1)."
            )
        if X.ndim != 2 and not allow_nd:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array")
    if X.dtype.kind == "f" and not np.all(np.isfinite(X.data if sp.issparse(X) else X)):
        raise ValueError("Input contains NaN or infinity")
    return X


def check_X_y(X, y, dtype=np.float64, accept_sparse=False, y_numeric=False):
    X = check_array(X, dtype=dtype, accept_sparse=accept_sparse)
    y = column_or_1d(y)
    n = X.shape[0]
    if y.shape[0] != n:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"[{n}, {y.shape[0]}]"
        )
    if y_numeric:
        y = y.astype(np.float64)
    return X, y


def column_or_1d(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y should be a 1d array, got shape {y.shape}")
    return y


def check_consistent_length(*arrays):
    lengths = {len(a) for a in arrays if a is not None}
    if len(lengths) > 1:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"{sorted(lengths)}"
        )


def unique_labels(*ys):
    return np.unique(np.concatenate([np.asarray(y).ravel() for y in ys]))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


__all__ = [
    "check_random_state", "check_array", "check_X_y", "column_or_1d",
    "check_consistent_length", "unique_labels", "softmax", "sigmoid",
]
