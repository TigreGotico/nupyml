"""Shape recognition from image descriptors, robust to POSITION and size.

Each image contains one shape (filled square, disk, or ring) at a random location
and scale on a noisy background. Because the shape moves, a model keying on
absolute pixels struggles; a descriptor that summarises local structure
(gradients, texture) generalises. The submission receives raw ``(n, 20, 20)``
images and extracts its own features. Scored by accuracy.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = "Classify shapes (square/disk/ring) at random position & scale; accuracy."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80


def _draw(shape, rng, S=20):
    img = 0.1 * rng.randn(S, S)
    r = rng.randint(4, 7)
    cy, cx = rng.randint(r, S - r), rng.randint(r, S - r)
    yy, xx = np.mgrid[0:S, 0:S]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if shape == 0:                                     # filled square
        img[np.abs(yy - cy) <= r] = img[np.abs(yy - cy) <= r]
        img[(np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)] += 1.0
    elif shape == 1:                                   # filled disk
        img[dist <= r] += 1.0
    else:                                              # ring
        img[(dist <= r) & (dist >= r - 1.5)] += 1.0
    return img


def load():
    rng = np.random.RandomState(0)
    X, y = [], []
    for cls in (0, 1, 2):
        for _ in range(150):
            X.append(_draw(cls, rng))
            y.append(cls)
    X, y = np.array(X), np.array(y)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr = int(0.7 * len(y))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
