"""Unsupervised anomaly detection: score how anomalous each point is.

The submission sees ONLY X -- no labels -- and must return a real-valued
anomaly score per point (higher = more anomalous). The hidden ground truth marks
which points are the injected outliers. Scored by ROC-AUC, which needs only the
RANKING of the scores, so a detector never has to guess a threshold.
"""
import numpy as np

from nupyml.metrics import roc_auc_score

KIND = "clustering"
GOAL = ("Rank points by how anomalous they are (unsupervised); scored by "
        "ROC-AUC against the hidden outlier labels.")
METRIC = "roc_auc"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.85


def load():
    rng = np.random.RandomState(0)
    d = 8
    # inliers: two dense gaussian clusters
    n_in = 950
    a = rng.randn(n_in // 2, d) + 3.0
    b = rng.randn(n_in - n_in // 2, d) - 3.0
    inliers = np.vstack([a, b])
    # outliers: spread uniformly across a much wider box, so they sit in the
    # sparse space between and around the clusters
    n_out = 50
    outliers = rng.uniform(-12, 12, size=(n_out, d))
    X = np.vstack([inliers, outliers])
    y = np.concatenate([np.zeros(n_in), np.ones(n_out)]).astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def metric(y_true, y_pred):
    # y_pred are anomaly SCORES (not labels); AUC ranks them against the truth
    return roc_auc_score(y_true, y_pred)
