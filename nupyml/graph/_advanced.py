"""Graph learning expansion: link prediction, kernels, flow, and LINE.

All functions take an adjacency matrix ``A`` (n x n; symmetric for an undirected
graph, 0/1 or weighted). They join the ``graph`` package's PageRank/HITS/
centrality/Louvain/node2vec/WL-kernel.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------

def common_neighbors(A):
    """Score a missing edge (i,j) by how many neighbours i and j share.

    The simplest link-prediction heuristic and the base of the others: two people
    with many mutual friends are likely to become friends. For a 0/1 adjacency,
    the count of common neighbours is exactly ``(A @ A)[i, j]``.
    """
    B = _binary(A)
    S = B @ B
    np.fill_diagonal(S, 0)
    return S


def jaccard_coefficient(A):
    """Common neighbours NORMALISED by the size of the combined neighbourhood.

    ``|N(i) ∩ N(j)| / |N(i) ∪ N(j)|`` -- so two low-degree nodes with a couple of
    shared neighbours can outrank two hubs that share many merely because they
    have many. The right correction when degrees vary a lot.
    """
    B = _binary(A)
    inter = B @ B
    deg = B.sum(axis=1)
    union = deg[:, None] + deg[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(S, 0)
    return S


def adamic_adar_index(A):
    """Common neighbours weighted by 1/log(degree): RARE shared friends count more.

    A mutual friend who knows everyone (a hub) is weak evidence you two are
    connected; a mutual friend with few links is strong evidence. Adamic-Adar
    down-weights each shared neighbour by ``1/log(deg)``, and it is one of the
    strongest simple predictors on social graphs.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    with np.errstate(divide="ignore"):
        w = np.where(deg > 1, 1.0 / np.log(deg), 0.0)
    S = (B * w[None, :]) @ B.T
    np.fill_diagonal(S, 0)
    return S


def resource_allocation_index(A):
    """Adamic-Adar's sibling with a 1/degree weight (a heavier penalty on hubs)."""
    B = _binary(A)
    deg = B.sum(axis=1)
    with np.errstate(divide="ignore"):
        w = np.where(deg > 0, 1.0 / deg, 0.0)
    S = (B * w[None, :]) @ B.T
    np.fill_diagonal(S, 0)
    return S


def preferential_attachment(A):
    """Score (i,j) by ``deg(i) * deg(j)`` -- the rich get richer.

    No shared-neighbour information at all: it bets that high-degree nodes keep
    attracting edges. Weak alone, but it captures the growth dynamics of many real
    networks and needs nothing but degrees.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    S = np.outer(deg, deg).astype(float)
    np.fill_diagonal(S, 0)
    return S


def katz_index(A, beta=0.01):
    """Sum over ALL paths between i and j, damped by length: (I - beta A)^-1 - I.

    Common neighbours only see length-2 paths; the Katz index counts paths of
    every length, weighting a length-``l`` path by ``beta^l`` so short paths
    dominate. ``beta`` must be below ``1/spectral-radius`` for the series to
    converge. It captures connectivity that shared-neighbour scores miss on sparse
    graphs, at the cost of a matrix inverse.
    """
    B = _binary(A)
    n = B.shape[0]
    S = np.linalg.inv(np.eye(n) - beta * B) - np.eye(n)
    np.fill_diagonal(S, 0)
    return S


# --- assortativity --------------------------------------------------------

def degree_assortativity(A):
    """Pearson correlation of the degrees at the two ends of an edge.

    Positive means high-degree nodes tend to link to other high-degree nodes
    (social networks); negative means hubs link to low-degree nodes (the
    technological/biological pattern). One number that captures a network's
    mixing character.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    iu = np.transpose(np.nonzero(np.triu(B, 1)))
    if len(iu) == 0:
        return 0.0
    x = deg[iu[:, 0]]
    y = deg[iu[:, 1]]
    # symmetric correlation: stack both edge orientations
    xs = np.concatenate([x, y])
    ys = np.concatenate([y, x])
    if xs.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


# --- Girvan-Newman community detection ------------------------------------

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

def shortest_path_kernel(A1, A2):
    """Compare two graphs by their distributions of shortest-path LENGTHS.

    Compute all-pairs shortest-path lengths in each graph; the kernel counts pairs
    of paths (one from each graph) with equal length. Two graphs that route
    information over similar distances score high, regardless of node identity --
    a permutation-invariant similarity. Floyd-Warshall gives the distances.
    """
    def apsp(A):
        B = _binary(A)
        n = B.shape[0]
        D = np.where(B > 0, 1.0, np.inf)
        np.fill_diagonal(D, 0)
        for k in range(n):
            D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
        return D
    d1 = apsp(A1)[np.triu_indices(A1.shape[0], 1)]
    d2 = apsp(A2)[np.triu_indices(A2.shape[0], 1)]
    d1 = d1[np.isfinite(d1)]; d2 = d2[np.isfinite(d2)]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    # count equal-length path pairs = dot product of length histograms
    maxlen = int(max(d1.max(), d2.max()))
    h1 = np.bincount(d1.astype(int), minlength=maxlen + 1)
    h2 = np.bincount(d2.astype(int), minlength=maxlen + 1)
    return float(h1 @ h2)


def random_walk_kernel(A1, A2, decay=0.1, max_len=10):
    """Count common walks in the two graphs via their DIRECT (tensor) product.

    A walk shared by both graphs is a walk in their product graph; summing over
    all lengths ``l`` with a ``decay^l`` weight gives the geometric-series kernel
    ``sum_l decay^l * 1' (A_x)^l 1`` on the product adjacency ``A_x = A1 ⊗ A2``.
    More walks in common -> more similar structure. Truncated at ``max_len``.
    """
    B1, B2 = _binary(A1), _binary(A2)
    Ax = np.kron(B1, B2)                             # product-graph adjacency
    n = Ax.shape[0]
    total = 0.0
    power = np.eye(n)
    for l in range(1, max_len + 1):
        power = power @ Ax
        total += (decay ** l) * power.sum()
    return float(total)


# --- LINE embedding -------------------------------------------------------

class LINE(BaseEstimator):
    """Large-scale Information Network Embedding (Tang et al., 2015).

    THE OBJECTIVE
    -------------
    Learn a vector per node so that CONNECTED nodes have similar embeddings --
    first-order proximity. LINE maximises, over the observed edges, the
    log-probability ``sigma(u_i · u_j)`` of the edge existing, against
    NEGATIVE-SAMPLED non-edges (random node pairs pushed apart). It is essentially
    node2vec's skip-gram objective applied directly to edges instead of to
    random-walk windows -- simpler, and designed to scale to millions of edges by
    sampling edges and negatives rather than materialising anything dense.

    (Second-order proximity -- sharing NEIGHBOURS -- uses a separate context
    embedding; this implements the first-order variant.)
    """

    def __init__(self, n_components=16, n_epochs=50, lr=0.025, n_negative=5,
                 random_state=None):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def fit(self, A):
        rng = check_random_state(self.random_state)
        B = _binary(A)
        n = B.shape[0]
        edges = np.transpose(np.nonzero(np.triu(B, 1)))
        self.embedding_ = rng.normal(0, 0.1, (n, self.n_components))
        U = self.embedding_
        for _ in range(self.n_epochs):
            rng.shuffle(edges)
            for i, j in edges:
                self._step(U, i, j, 1.0)             # positive edge: pull together
                for _ in range(self.n_negative):
                    k = rng.randint(n)               # negative: push apart
                    self._step(U, i, k, 0.0)
        return self

    def _step(self, U, i, j, label):
        score = 1.0 / (1.0 + np.exp(-np.clip(U[i] @ U[j], -30, 30)))
        g = self.lr * (label - score)
        ui = U[i].copy()
        U[i] += g * U[j]
        U[j] += g * ui

    def fit_transform(self, A):
        return self.fit(A).embedding_


__all__ = ["common_neighbors", "jaccard_coefficient", "adamic_adar_index",
           "resource_allocation_index", "preferential_attachment", "katz_index",
           "degree_assortativity", "girvan_newman", "max_flow", "min_cut",
           "shortest_path_kernel", "random_walk_kernel", "LINE"]
