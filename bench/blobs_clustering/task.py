"""Blobs clustering: recover the generating clusters WITHOUT labels.

An unsupervised task: the submission sees only X and must produce a label per
point. Scored by the adjusted Rand index against the hidden ground-truth
assignment, which is invariant to how the clusters are numbered.
"""
from nupyml.datasets import make_blobs
from nupyml.metrics import adjusted_rand_score

KIND = "clustering"
GOAL = "Cluster points into their generating blobs (unsupervised); scored by ARI."
METRIC = "adjusted_rand_index"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.70


def load():
    # five well-separated blobs; the true labels are held out for scoring only
    X, y = make_blobs(n_samples=1000, n_features=5, centers=5, cluster_std=1.2,
                      random_state=0)
    return X, y


def metric(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
