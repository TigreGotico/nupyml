"""Point-cloud classification: label an unordered 3-D point set by its shape.

Each sample is a cloud of 3-D points sampled from either a sphere or a cube surface,
randomly rotated. A classifier must be invariant to point ORDER and to rotation.
Scored by accuracy. Showcases PointNet against a hand-crafted bag-of-points baseline.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = "Classify unordered 3-D point clouds (sphere vs cube surface); accuracy."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80


def _random_rotation(rng):
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return Rz @ Ry @ Rx


def _cloud(kind, n_points, rng):
    if kind == 0:                                          # sphere surface
        v = rng.randn(n_points, 3)
        pts = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    else:                                                  # cube surface
        pts = rng.uniform(-1, 1, (n_points, 3))
        face = rng.randint(0, 3, n_points)
        sign = rng.choice([-1, 1], n_points)
        pts[np.arange(n_points), face] = sign             # pin one coord to a face
    return pts @ _random_rotation(rng).T


def load():
    rng = np.random.RandomState(0)
    n_per, n_points = 200, 32
    clouds, labels = [], []
    for k in (0, 1):
        for _ in range(n_per):
            clouds.append(_cloud(k, n_points, rng)); labels.append(k)
    X = np.array(clouds); y = np.array(labels)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
