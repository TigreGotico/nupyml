"""Divisive community detection: repeatedly cut the most 'between' edge."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def _edge_betweenness(B):
    """Brandes-style accumulation of betweenness onto EDGES (unweighted BFS)."""
    n = B.shape[0]
    eb = np.zeros((n, n))
    neighbors = [np.nonzero(B[v])[0] for v in range(n)]
    for s in range(n):
        # BFS from s
        S, P = [], [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1
        dist = -np.ones(n); dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            S.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        for w in reversed(S):
            for v in P[w]:
                c = (sigma[v] / sigma[w]) * (1 + delta[w])
                eb[v, w] += c
                eb[w, v] += c
                delta[v] += c
    return eb / 2.0


def girvan_newman(A, n_communities=2):
    """Divisive community detection: repeatedly cut the most 'between' edge.

    THE IDEA
    --------
    The edges that BRIDGE communities carry the most shortest paths between them,
    so they have high edge betweenness. Girvan-Newman removes the
    highest-betweenness edge, recomputes, and repeats; the graph fragments into
    communities from the outside in. Stop when it has split into
    ``n_communities`` connected components.

    It is the interpretable, top-down counterpart to Louvain's bottom-up
    optimisation -- slower (betweenness is recomputed each cut) but it exposes the
    bridging edges themselves, which is often what you want to SEE.

    Girvan & Newman (2002). Returns a label per node.
    """
    B = _binary(A).copy()
    np.fill_diagonal(B, 0)

    def components(M):
        n = M.shape[0]
        lab = -np.ones(n, dtype=int)
        c = 0
        for start in range(n):
            if lab[start] >= 0:
                continue
            stack = [start]; lab[start] = c
            while stack:
                v = stack.pop()
                for w in np.nonzero(M[v])[0]:
                    if lab[w] < 0:
                        lab[w] = c; stack.append(w)
            c += 1
        return lab, c

    lab, c = components(B)
    while c < n_communities and B.sum() > 0:
        eb = _edge_betweenness(B)
        i, j = np.unravel_index(np.argmax(eb), eb.shape)
        B[i, j] = B[j, i] = 0                        # cut the bridge
        lab, c = components(B)
    return lab


# --- max-flow / min-cut ---------------------------------------------------


__all__ = ["girvan_newman"]
