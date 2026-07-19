"""Hyperbolic hierarchy: embed a tree so graph distances are preserved.

The inputs are node pairs from a balanced tree; the target is each pair's tree (hop)
distance. A submission must embed the nodes -- from the training pairs alone -- and
predict the distances of held-out pairs. Trees need hyperbolic space to embed with low
distortion, so this rewards a hyperbolic embedding over a Euclidean one. Scored by the
Spearman rank correlation between predicted and true distances. Framed as supervised.
"""
import numpy as np
from scipy.stats import spearmanr

KIND = "supervised"
GOAL = "Embed a tree and predict held-out pairwise graph distances; Spearman rho."
METRIC = "spearman_rho"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.55


def _balanced_tree_edges(depth=4, branching=2):
    edges = []; nxt = 1; frontier = [0]
    while frontier and nxt < branching ** (depth + 1):
        node = frontier.pop(0)
        for _ in range(branching):
            edges.append((node, nxt)); frontier.append(nxt); nxt += 1
            if nxt >= (branching ** (depth + 1) - 1) // (branching - 1):
                break
    return edges, nxt


def _all_pairs_hops(edges, n):
    import collections
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    D = np.full((n, n), -1)
    for s in range(n):
        D[s, s] = 0; q = collections.deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[s, w] < 0:
                    D[s, w] = D[s, u] + 1; q.append(w)
    return D


def load():
    rng = np.random.RandomState(0)
    edges, n = _balanced_tree_edges(depth=4)
    D = _all_pairs_hops(edges, n)
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if (i, j) not in edge_set]
    rng.shuffle(non_edges)
    cut = len(non_edges) // 2
    # training carries EVERY edge (so the graph is recoverable) + half the non-edges
    train_pairs = list(edge_set) + non_edges[:cut]
    test_pairs = non_edges[cut:]
    Xtr = np.array(train_pairs); ytr = np.array([D[i, j] for i, j in train_pairs])
    Xte = np.array(test_pairs); yte = np.array([D[i, j] for i, j in test_pairs])
    return Xtr, ytr, Xte, yte


def metric(y_true, y_pred):
    rho = spearmanr(y_true, np.asarray(y_pred)).correlation
    return float(rho if np.isfinite(rho) else 0.0)
