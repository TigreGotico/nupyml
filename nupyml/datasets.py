"""Datasets: synthetic generators and a few small real ones.

THE GENERATORS
--------------
Each ``make_*`` function produces data with a KNOWN structure, which makes them
the right tool for understanding an algorithm's assumptions -- you know the
answer, so you can see exactly how and where a method fails:

* ``make_blobs``     -- gaussian clusters. What KMeans assumes.
* ``make_moons``     -- two interleaving crescents. Not linearly separable, and
  not what KMeans assumes: it will cut them in half. DBSCAN and spectral
  clustering handle them.
* ``make_circles``   -- concentric rings. Defeats every centroid method; the
  standard demonstration of why the kernel trick and spectral methods exist.
* ``make_classification`` / ``make_regression`` -- controllable numbers of
  informative and noise features, for studying regularisation and selection.
* ``make_spiral``    -- non-linear and non-convex; a small hard problem.

THE BUNDLED DATA
----------------
``load_iris``, ``load_wine``, ``load_breast_cancer``, ``load_digits``,
``load_diabetes`` and ``load_linnerud`` are the classic small UCI sets, stored
in the package so nothing is downloaded and no extra dependency is needed. They
are small enough to experiment on instantly and real enough to be interesting.
"""
import os

import numpy as np

from .utils import check_random_state

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Bunch(dict):
    """A dict whose keys are also attributes."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _load_bundled(name, return_X_y=False):
    path = os.path.join(_DATA_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"bundled dataset {name!r} is missing at {path}")
    with np.load(path, allow_pickle=True) as f:
        data, target = f["data"], f["target"]
        feature_names = [str(x) for x in f["feature_names"]]
        target_names = np.array([str(x) for x in f["target_names"]])
    if return_X_y:
        return data, target
    return Bunch(data=data, target=target, feature_names=feature_names,
                 target_names=target_names)


def load_iris(return_X_y=False):
    """150 iris flowers, 4 measurements, 3 species."""
    return _load_bundled("iris", return_X_y)


def load_wine(return_X_y=False):
    """178 wines, 13 chemical measurements, 3 cultivars."""
    return _load_bundled("wine", return_X_y)


def load_breast_cancer(return_X_y=False):
    """569 tumours, 30 features, malignant/benign."""
    return _load_bundled("breast_cancer", return_X_y)


def load_digits(return_X_y=False):
    """1797 8x8 handwritten digit images, 10 classes."""
    bunch = _load_bundled("digits", return_X_y)
    if return_X_y:
        return bunch
    bunch.images = bunch.data.reshape(-1, 8, 8)
    return bunch


def load_diabetes(return_X_y=False):
    """442 patients, 10 baseline features, disease progression target."""
    return _load_bundled("diabetes", return_X_y)


def load_linnerud(return_X_y=False):
    """20 athletes: exercise measurements and physiological targets."""
    return _load_bundled("linnerud", return_X_y)


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), random_state=None):
    rng = check_random_state(random_state)
    if isinstance(centers, int):
        centers = rng.uniform(center_box[0], center_box[1], size=(centers, n_features))
    else:
        centers = np.asarray(centers, dtype=np.float64)
        n_features = centers.shape[1]
    n_centers = len(centers)
    counts = np.full(n_centers, n_samples // n_centers)
    counts[: n_samples % n_centers] += 1
    X, y = [], []
    stds = np.broadcast_to(np.asarray(cluster_std, dtype=np.float64).ravel(),
                           (n_centers,)) if np.ndim(cluster_std) else \
        np.full(n_centers, cluster_std)
    for i, (c, n_i) in enumerate(zip(centers, counts)):
        X.append(rng.normal(c, stds[i], size=(n_i, n_features)))
        y.append(np.full(n_i, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_moons(n_samples=100, noise=None, random_state=None):
    rng = check_random_state(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = np.linspace(0, np.pi, n_out)
    theta_in = np.linspace(0, np.pi, n_in)
    X = np.vstack([
        np.column_stack([np.cos(theta_out), np.sin(theta_out)]),
        np.column_stack([1 - np.cos(theta_in), 1 - np.sin(theta_in) - 0.5]),
    ])
    y = np.r_[np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)]
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_circles(n_samples=100, noise=None, factor=0.8, random_state=None):
    rng = check_random_state(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    theta_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    X = np.vstack([
        np.column_stack([np.cos(theta_out), np.sin(theta_out)]),
        factor * np.column_stack([np.cos(theta_in), np.sin(theta_in)]),
    ])
    y = np.r_[np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)]
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_classes=2, class_sep=1.0, random_state=None):
    rng = check_random_state(random_state)
    counts = np.full(n_classes, n_samples // n_classes)
    counts[: n_samples % n_classes] += 1
    centers = rng.normal(scale=class_sep * 2, size=(n_classes, n_informative))
    X_inf, y = [], []
    for i, n_i in enumerate(counts):
        X_inf.append(rng.normal(centers[i], 1.0, size=(n_i, n_informative)))
        y.append(np.full(n_i, i))
    X_inf = np.vstack(X_inf)
    y = np.concatenate(y)
    n_noise = n_features - n_informative
    X = np.hstack([X_inf, rng.normal(size=(n_samples, n_noise))]) if n_noise > 0 else X_inf
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.0,
                    coef=False, random_state=None):
    rng = check_random_state(random_state)
    X = rng.normal(size=(n_samples, n_features))
    w = np.zeros(n_features)
    informative = rng.choice(n_features, size=min(n_informative, n_features), replace=False)
    w[informative] = rng.uniform(10, 100, size=len(informative))
    y = X @ w + rng.normal(scale=noise, size=n_samples)
    if coef:
        return X, y, w
    return X, y


def make_spiral(n_samples=100, n_arms=2, noise=0.1, random_state=None):
    rng = check_random_state(random_state)
    per = n_samples // n_arms
    X, y = [], []
    for arm in range(n_arms):
        t = np.linspace(0.5, 3.0, per)
        angle = t * 2 * np.pi / n_arms + arm * 2 * np.pi / n_arms
        X.append(np.column_stack([t * np.cos(angle * 2), t * np.sin(angle * 2)])
                 + rng.normal(scale=noise, size=(per, 2)))
        y.append(np.full(per, arm))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


__all__ = ["make_blobs", "make_moons", "make_circles", "make_classification",
           "make_regression", "make_spiral", "Bunch", "load_iris", "load_wine",
           "load_breast_cancer", "load_digits", "load_diabetes",
           "load_linnerud"]
