"""Node-importance algorithms: PageRank, HITS, and classical centralities."""
import numpy as np


def pagerank(A, damping=0.85, max_iter=200, tol=1e-9):
    """PageRank: a node is important if IMPORTANT nodes link to it.

    THE RECURSIVE IDEA
    ------------------
    Importance is defined circularly -- yours depends on your linkers', theirs on
    their linkers' -- and the fixed point of that recursion is the answer. Model a
    random surfer who follows a link with probability ``damping`` and teleports to
    a random node with probability ``1 - damping``; PageRank is the fraction of
    time spent at each node in the long run, i.e. the STATIONARY DISTRIBUTION of
    that walk. Solved by power iteration: start uniform, repeatedly push each
    node's score along its out-links, until it settles.

    THE DAMPING FACTOR
    ------------------
    The teleport term (``1 - damping``, classically 0.15) is not a detail -- it is
    what makes the walk well-posed. It guarantees the chain is irreducible and
    aperiodic (so a unique stationary distribution exists) and rescues "rank
    sinks": a node or group with no out-links would otherwise absorb all the
    probability. Teleporting leaks it back out. This is the algorithm that made
    Google, and it is exactly the ``MarkovChain`` stationary distribution from
    ``pgm`` applied to the link matrix.
    """
    A = np.asarray(A, float)
    n = len(A)
    # column-stochastic transition: each node splits its score among its links
    out_deg = A.sum(axis=1, keepdims=True)
    # dangling nodes (no out-links) teleport everywhere -- the rank-sink fix
    dangling = (out_deg.ravel() == 0)
    M = np.where(out_deg > 0, A / np.where(out_deg == 0, 1, out_deg), 0.0).T

    r = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        # dangling mass is redistributed uniformly, then damping + teleport
        r_new = damping * (M @ r + dangling @ r / n) + (1 - damping) * teleport
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    return r / r.sum()


def hits(A, max_iter=200, tol=1e-9):
    """HITS: split importance into HUBS and AUTHORITIES.

    THE TWO ROLES
    -------------
    A node can matter in two ways. An AUTHORITY is pointed to by good hubs (a
    definitive page many good directories link to); a HUB points to good
    authorities (a good directory). The two reinforce each other: a good hub links
    to good authorities, a good authority is linked by good hubs. Iterate that
    mutual definition -- authority score = sum of linking hubs' scores, hub score
    = sum of linked authorities' scores -- and both converge (they are the
    singular vectors of ``A``).

    The contrast with PageRank: PageRank gives one score per node from a random
    walk; HITS gives TWO, separating "good source" from "good target", which is
    the right decomposition for directed, hub-and-spoke structure like the web or
    citation graphs.

    Kleinberg (1999). Returns ``(hubs, authorities)``.
    """
    A = np.asarray(A, float)
    n = len(A)
    hubs = np.ones(n)
    auth = np.ones(n)
    for _ in range(max_iter):
        # authority = sum of incoming hub scores; hub = sum of outgoing authority
        auth_new = A.T @ hubs
        auth_new /= np.linalg.norm(auth_new) + 1e-12
        hubs_new = A @ auth_new
        hubs_new /= np.linalg.norm(hubs_new) + 1e-12
        if np.max(np.abs(auth_new - auth)) < tol:
            auth, hubs = auth_new, hubs_new
            break
        auth, hubs = auth_new, hubs_new
    return hubs, auth


def centrality(A, kind="degree"):
    """Classical node centrality -- four different meanings of "central".

    * ``degree`` -- how many neighbours. Local popularity; cheap and often enough.
    * ``closeness`` -- inverse of the mean shortest-path distance to everyone.
      "How quickly can this node reach the rest" -- central = near everything.
    * ``betweenness`` -- fraction of all shortest paths that pass THROUGH the
      node. "How much of the traffic must go through me" -- finds bridges and
      bottlenecks, which degree misses entirely (a low-degree node joining two
      clusters has huge betweenness).
    * ``eigenvector`` -- a node is central if its NEIGHBOURS are central (the
      recursive idea, and PageRank's undirected ancestor).

    The point of offering all four: they disagree, and which one is "right"
    depends entirely on what "important" means for the problem -- popularity,
    reach, brokerage, or influence.
    """
    A = np.asarray(A, float)
    n = len(A)
    if kind == "degree":
        return A.sum(axis=1) / (n - 1)              # normalised degree
    if kind == "eigenvector":
        vals, vecs = np.linalg.eig(A)
        principal = np.abs(np.real(vecs[:, np.argmax(np.real(vals))]))
        return principal / principal.sum()
    if kind in ("closeness", "betweenness"):
        dist, paths = _all_pairs_shortest(A)
        if kind == "closeness":
            out = np.zeros(n)
            for i in range(n):
                reachable = dist[i][np.isfinite(dist[i]) & (dist[i] > 0)]
                if len(reachable):
                    out[i] = len(reachable) / reachable.sum()
            return out
        return _betweenness(A, dist)
    raise ValueError(f"unknown centrality kind {kind!r}")


def _all_pairs_shortest(A):
    """BFS from every node on the unweighted graph (edges where A>0)."""
    n = len(A)
    adj = [np.where(A[i] > 0)[0] for i in range(n)]
    dist = np.full((n, n), np.inf)
    for s in range(n):
        dist[s, s] = 0
        queue = [s]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1
                    queue.append(v)
    return dist, adj


def _betweenness(A, dist):
    """Fraction of shortest paths through each node (Brandes-style, unweighted).

    Counts, over all source-target pairs, how many shortest paths route through
    each intermediate node -- the exact structural definition of a broker.
    """
    n = len(A)
    adj = [np.where(A[i] > 0)[0] for i in range(n)]
    bc = np.zeros(n)
    for s in range(n):
        # Brandes: BFS, count shortest paths, then accumulate dependencies back
        stack, pred = [], [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1
        d = np.full(n, -1); d[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj[v]:
                if d[w] < 0:
                    d[w] = d[v] + 1
                    queue.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    # undirected graph: each path counted twice
    norm = (n - 1) * (n - 2)
    return bc / norm if norm > 0 else bc


__all__ = ["pagerank", "hits", "centrality"]
