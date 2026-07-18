"""Time-series anomaly detection: flag the unusual timesteps.

A smooth seasonal series is corrupted at a few timesteps with spikes and level jumps.
Each timestep is described by local features (its value, deviation from a rolling
mean, and local volatility), and the task -- unsupervised -- is to score how anomalous
each point is so the injected anomalies rank above the normal ones. Scored by ROC-AUC
against the hidden anomaly labels.
"""
import numpy as np

from nupyml.metrics import roc_auc_score

KIND = "clustering"
GOAL = "Score timesteps by how anomalous they are; ROC-AUC vs injected anomalies."
METRIC = "roc_auc"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.6


def load():
    rng = np.random.RandomState(0)
    n = 500
    t = np.arange(n)
    y = 5 * np.sin(2 * np.pi * t / 40) + 0.3 * rng.randn(n)
    labels = np.zeros(n, int)
    anom = rng.choice(np.arange(20, n - 20), 20, replace=False)
    for a in anom:
        y[a] += rng.choice([-1, 1]) * rng.uniform(4, 7)   # spikes
        labels[a] = 1
    w = 15
    roll_mean = np.array([y[max(0, i - w):i + 1].mean() for i in range(n)])
    roll_std = np.array([y[max(0, i - w):i + 1].std() + 1e-6 for i in range(n)])
    deviation = np.abs(y - roll_mean)
    X = np.column_stack([y, deviation, deviation / roll_std])
    return X, labels


def metric(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))
