"""Multi-label classification: predict a SET of labels per example.

Each sample can carry several labels at once, and the labels are CORRELATED
(some tend to co-occur). A method that models those correlations -- rather than
predicting each label independently -- should win. Scored by macro-averaged F1
over the label matrix.
"""
import numpy as np

KIND = "supervised"
GOAL = ("Predict a set of correlated labels per sample; scored by "
        "macro-averaged F1 over the label columns.")
METRIC = "macro_f1_multilabel"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.55


def load():
    rng = np.random.RandomState(0)
    n, d, n_labels = 600, 20, 5
    X = rng.randn(n, d)
    # each label is a logistic function of a random projection...
    W = rng.randn(d, n_labels)
    logits = X @ W
    # ...plus shared latent factors that make labels co-occur (correlation)
    shared = rng.randn(n, 2) @ rng.randn(2, n_labels) * 1.5
    prob = 1.0 / (1.0 + np.exp(-(logits + shared)))
    Y = (rng.rand(n, n_labels) < prob).astype(int)
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]
    n_tr = int(0.7 * n)
    return X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]


def metric(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for j in range(y_true.shape[1]):
        tp = np.sum((y_pred[:, j] == 1) & (y_true[:, j] == 1))
        fp = np.sum((y_pred[:, j] == 1) & (y_true[:, j] == 0))
        fn = np.sum((y_pred[:, j] == 0) & (y_true[:, j] == 1))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))
