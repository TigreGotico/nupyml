"""Baseline: reconstruct the graph from unit-distance pairs and use BFS distances."""
import collections

import numpy as np


def solve(X_train, y_train, X_test):
    n = int(max(X_train.max(), X_test.max())) + 1
    adj = collections.defaultdict(list)
    for (u, v), d in zip(X_train, y_train):
        if d == 1:                                         # the tree edges
            adj[u].append(v); adj[v].append(u)
    D = np.full((n, n), n + 1)
    for s in range(n):
        D[s, s] = 0; q = collections.deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[s, w] == n + 1:
                    D[s, w] = D[s, u] + 1; q.append(w)
    return np.array([D[u, v] for u, v in X_test], dtype=float)
