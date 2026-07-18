"""Image segmentation: group pixels into the regions they belong to.

A synthetic image is built from a few smooth colour regions plus noise. Each pixel
becomes a feature vector of its (row, col) position and intensity, and the task is
to cluster the pixels back into their regions -- an unsupervised segmentation.
Scored by the adjusted Rand index against the true region map, so both over- and
under-segmentation are penalised.
"""
import numpy as np

from nupyml.metrics import adjusted_rand_score

KIND = "clustering"
GOAL = "Segment an image's pixels into regions; adjusted Rand index."
METRIC = "adjusted_rand"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.55


def load():
    rng = np.random.RandomState(0)
    H = W = 24
    labels = np.zeros((H, W), int)
    intensity = np.zeros((H, W))
    # four quadrants, each a distinct intensity level
    for qi, (ys, xs, val) in enumerate([((0, 12), (0, 12), 0.2),
                                        ((0, 12), (12, 24), 0.5),
                                        ((12, 24), (0, 12), 0.8),
                                        ((12, 24), (12, 24), 0.35)]):
        labels[ys[0]:ys[1], xs[0]:xs[1]] = qi
        intensity[ys[0]:ys[1], xs[0]:xs[1]] = val
    intensity += rng.randn(H, W) * 0.04
    rows, cols = np.mgrid[0:H, 0:W]
    X = np.column_stack([rows.ravel() / H, cols.ravel() / W,
                         intensity.ravel() * 2.0])       # (n_pixels, 3)
    y = labels.ravel()
    return X, y


def metric(y_true, y_pred):
    return float(adjusted_rand_score(y_true, y_pred))
