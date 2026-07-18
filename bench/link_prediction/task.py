"""Link prediction: will an edge form between two nodes?

A community graph (stochastic block model) is built, then some edges are hidden.
Each candidate node pair is described by classic link-prediction scores -- common
neighbours, Jaccard, Adamic-Adar, preferential attachment -- and the task is to
tell true (held-out) edges from non-edges. Scored by ROC-AUC over the test pairs,
so it rewards ranking real links above spurious ones.
"""
import numpy as np

from nupyml.metrics import roc_auc_score

KIND = "supervised"
GOAL = "Classify node pairs as linked / not from graph features; ROC-AUC."
METRIC = "roc_auc"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.75


def _features(A, pairs):
    deg = A.sum(axis=1)
    feats = []
    for i, j in pairs:
        common = np.where((A[i] > 0) & (A[j] > 0))[0]
        union = np.where((A[i] > 0) | (A[j] > 0))[0]
        cn = len(common)
        jac = cn / max(len(union), 1)
        aa = np.sum(1.0 / np.log(deg[common] + 1e-9)) if len(common) else 0.0
        pa = deg[i] * deg[j]
        feats.append([cn, jac, aa, pa])
    return np.array(feats, float)


def load():
    rng = np.random.RandomState(0)
    sizes = [30, 30, 30]
    n = sum(sizes)
    labels = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    A = np.zeros((n, n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.55 if labels[i] == labels[j] else 0.02
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
                edges.append((i, j))
    edges = np.array(edges)
    rng.shuffle(edges)
    # hide 30% of edges as positives; sample equal negatives (non-edges)
    n_hide = len(edges) // 3
    hidden = edges[:n_hide]
    A_obs = A.copy()
    for i, j in hidden:
        A_obs[i, j] = A_obs[j, i] = 0
    non_edges = []
    while len(non_edges) < len(edges):
        i, j = rng.randint(n, size=2)
        if i != j and A[i, j] == 0:
            non_edges.append((i, j))
    non_edges = np.array(non_edges)
    pos, neg = edges, non_edges
    X = np.vstack([_features(A_obs, pos), _features(A_obs, neg)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def metric(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))
