"""Node classification: label graph nodes from their connectivity.

A graph is built from three communities (stochastic block model): nodes connect
densely WITHIN their community and sparsely across. Each node's feature vector is
its row of the adjacency matrix -- who it links to -- and the label is its
community. A classifier (or a submission that first embeds the nodes) must recover
the communities from connectivity alone. Accuracy on held-out nodes, higher is
better.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = "Classify graph nodes into communities from adjacency features; accuracy."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.7


def load():
    rng = np.random.RandomState(0)
    sizes = [40, 40, 40]
    n = sum(sizes)
    labels = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    p_in, p_out = 0.30, 0.07
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    idx = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = idx[:split], idx[split:]
    return A[tr], labels[tr], A[te], labels[te]


def metric(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))
