"""Structure learning: discover the graph itself from data.

Given only samples, which variables depend directly on which? This is harder
than fitting the tables -- the space of graphs is super-exponential -- so the two
classic answers trade generality for tractability: Chow-Liu finds the best TREE
exactly, hill-climbing searches general DAGs greedily.
"""
import numpy as np


def _mutual_information(X, i, j, card_i, card_j):
    """Empirical mutual information between two columns -- how much knowing one
    reduces uncertainty about the other. Zero iff they are independent."""
    n = len(X)
    joint = np.zeros((card_i, card_j))
    for a, b in zip(X[:, i], X[:, j]):
        joint[a, b] += 1
    joint /= n
    pi = joint.sum(axis=1, keepdims=True)
    pj = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask]
                                             / (pi @ pj)[mask])))


def chow_liu(X, cardinalities=None):
    """The optimal tree-structured model, in closed form.

    THE RESULT
    ----------
    Among ALL tree structures, the one that best approximates the true joint
    (minimum KL divergence) is the MAXIMUM-WEIGHT SPANNING TREE where each edge is
    weighted by the mutual information between its two variables. This is exact
    and cheap -- no search -- which is why Chow-Liu (1968) is still the go-to when
    a tree is an acceptable approximation.

    The intuition: a tree can keep ``n-1`` dependencies, so keep the STRONGEST
    ``n-1`` (highest mutual information) that still form a tree. Kruskal's
    algorithm on the negated MI weights does exactly that.

    Returns a list of undirected edges (variable-index pairs).
    """
    X = np.asarray(X)
    n, d = X.shape
    if cardinalities is None:
        cardinalities = [int(X[:, j].max()) + 1 for j in range(d)]

    # all pairwise mutual informations = candidate edge weights
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            mi = _mutual_information(X, i, j, cardinalities[i], cardinalities[j])
            edges.append((mi, i, j))
    edges.sort(reverse=True)            # strongest dependency first

    # Kruskal: add the heaviest edge that does not form a cycle
    parent = list(range(d))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for mi, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
            if len(tree) == d - 1:      # a spanning tree has exactly n-1 edges
                break
    return tree


def hill_climb_structure(X, cardinalities=None, max_parents=3, max_iter=100):
    """Greedy DAG search by a penalised-likelihood score (BIC).

    THE SEARCH
    ----------
    The space of DAGs is too large to enumerate, so hill-climbing starts from the
    empty graph and repeatedly applies the single edge operation -- add, remove,
    or reverse -- that most improves the score, stopping at a local optimum. It is
    the standard practical structure learner, and like any greedy search it can
    get stuck, so multiple restarts help.

    THE SCORE
    ---------
    BIC = log-likelihood minus a complexity penalty
    ``0.5 * (#parameters) * log(n)``. The penalty is essential: without it the
    fully-connected graph always wins (more edges never lower the likelihood), so
    the search would just connect everything. BIC asks each edge to EARN its
    parameters, which is what makes the learned structure sparse and meaningful.

    Every candidate is checked for acyclicity -- a DAG that gained a cycle is no
    longer a valid Bayes net.

    Returns a list of directed edges (parent-index, child-index).
    """
    from .bayes_net import BayesianNetwork
    X = np.asarray(X)
    n, d = X.shape
    if cardinalities is None:
        cardinalities = [int(X[:, j].max()) + 1 for j in range(d)]
    card = {j: cardinalities[j] for j in range(d)}

    def score(edges):
        net = BayesianNetwork(edges=edges).fit(
            X, columns=list(range(d)), cardinalities=card)
        ll = net.log_likelihood(X)
        # count free parameters: for each node, (card-1) per parent configuration
        n_params = 0
        parents = {j: [] for j in range(d)}
        for p, c in edges:
            parents[c].append(p)
        for j in range(d):
            pa_prod = int(np.prod([card[p] for p in parents[j]])) if parents[j] else 1
            n_params += (card[j] - 1) * pa_prod
        return ll - 0.5 * n_params * np.log(n)

    def has_cycle(edges):
        adj = {j: [] for j in range(d)}
        for p, c in edges:
            adj[p].append(c)
        color = {j: 0 for j in range(d)}      # 0=unseen 1=in-stack 2=done

        def dfs(u):
            color[u] = 1
            for v in adj[u]:
                if color[v] == 1 or (color[v] == 0 and dfs(v)):
                    return True
            color[u] = 2
            return False
        return any(color[j] == 0 and dfs(j) for j in range(d))

    edges = []
    current = score(edges)
    for _ in range(max_iter):
        best_edges, best_score = edges, current
        # every legal single-edge move from the current graph
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if (i, j) in edges:                 # remove
                    cand = [e for e in edges if e != (i, j)]
                elif (j, i) in edges:               # reverse
                    cand = [e for e in edges if e != (j, i)] + [(i, j)]
                else:                               # add
                    n_pa = sum(1 for p, c in edges if c == j)
                    if n_pa >= max_parents:
                        continue
                    cand = edges + [(i, j)]
                if has_cycle(cand):
                    continue
                s = score(cand)
                if s > best_score:
                    best_score, best_edges = s, cand
        if best_score <= current:                   # local optimum reached
            break
        edges, current = best_edges, best_score
    return edges


__all__ = ["chow_liu", "hill_climb_structure"]
