"""Edmonds-Karp: maximum flow from source to sink (BFS-augmenting-paths)."""
import numpy as np


def max_flow(capacity, source, sink):
    """Edmonds-Karp: maximum flow from source to sink (BFS-augmenting-paths).

    Returns (flow_value, flow_matrix). The max-flow min-cut theorem says this
    value also equals the minimum total capacity that must be cut to disconnect
    source from sink -- so ``min_cut`` reads the answer off the residual graph.
    """
    C = np.array(capacity, dtype=float)
    n = C.shape[0]
    F = np.zeros((n, n))
    while True:
        # BFS for an augmenting path in the residual graph
        parent = -np.ones(n, dtype=int); parent[source] = source
        queue = [source]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if parent[v] < 0 and C[u, v] - F[u, v] > 1e-12:
                    parent[v] = u; queue.append(v)
        if parent[sink] < 0:
            break                                    # no path -> done
        # bottleneck along the path
        path, v = [], sink
        while v != source:
            path.append((parent[v], v)); v = parent[v]
        bottleneck = min(C[u, w] - F[u, w] for u, w in path)
        for u, w in path:
            F[u, w] += bottleneck
            F[w, u] -= bottleneck
    return float(F[source].sum()), F

def min_cut(capacity, source, sink):
    """The minimum s-t cut: (cut_value, source_side_nodes). Reachable-in-residual
    nodes form the source side; edges crossing to the rest are the cut."""
    value, F = max_flow(capacity, source, sink)
    C = np.array(capacity, dtype=float)
    n = C.shape[0]
    reachable = np.zeros(n, dtype=bool); reachable[source] = True
    queue = [source]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not reachable[v] and C[u, v] - F[u, v] > 1e-12:
                reachable[v] = True; queue.append(v)
    return value, np.nonzero(reachable)[0]


# --- graph kernels --------------------------------------------------------


__all__ = ["max_flow", "min_cut"]
