"""Community detection: finding groups of densely-connected nodes."""
import numpy as np

from ..utils import check_random_state


def modularity(A, labels):
    """How much better-than-random the given partition is.

    Modularity compares the fraction of edges INSIDE communities to what you would
    expect if the same-degree nodes were wired up at random::

        Q = (1/2m) sum_ij [A_ij - k_i k_j / 2m] * [same community?]

    A high Q means the partition captures real density -- many more within-group
    edges than chance. It is the objective community detection maximises, and the
    yardstick for comparing partitions. Values above ~0.3 usually indicate genuine
    community structure.
    """
    A = np.asarray(A, float)
    labels = np.asarray(labels)
    m = A.sum() / 2
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    same = labels[:, None] == labels[None, :]
    # observed minus expected edge fraction, summed within communities
    Q = np.sum((A - np.outer(k, k) / (2 * m)) * same) / (2 * m)
    return float(Q)


def louvain(A, max_iter=100, random_state=None):
    """Greedy modularity maximisation -- the standard community detector.

    THE ALGORITHM
    -------------
    Two phases repeated. LOCAL MOVING: with every node its own community, visit
    nodes in turn and move each into the neighbouring community that most raises
    modularity, until no move helps. AGGREGATION: collapse each community into a
    single super-node (edges become weighted) and recurse on the smaller graph.
    Each pass builds coarser communities, and the number of communities is
    DISCOVERED, not specified -- unlike k-means, you do not say how many.

    It is greedy and fast (near-linear in practice), which is why Louvain scales
    to graphs with millions of nodes where exact modularity maximisation (NP-hard)
    is hopeless. The trade is the usual greedy caveat: it finds a strong local
    optimum, not necessarily the global one, and can occasionally produce slightly
    disconnected communities -- which is what the Leiden refinement later fixed.

    Blondel, Guillaume, Lambiotte & Lefebvre (2008).
    """
    A = np.asarray(A, float)
    rng = check_random_state(random_state)
    n = len(A)
    labels = np.arange(n)               # every node starts alone
    m = A.sum() / 2
    if m == 0:
        return labels

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in rng.permutation(n):
            # try moving node i into each neighbour's community; keep the best gain
            neighbours = np.where(A[i] > 0)[0]
            if len(neighbours) == 0:
                continue
            current = labels[i]
            best_gain, best_label = 0.0, current
            candidate_labels = set(labels[neighbours].tolist())
            for lab in candidate_labels:
                if lab == current:
                    continue
                # modularity gain from moving i into community `lab`
                labels[i] = lab
                gain = _local_gain(A, labels, i, m)
                labels[i] = current
                if gain > best_gain:
                    best_gain, best_label = gain, lab
            if best_label != current:
                labels[i] = best_label
                improved = True

    # relabel to a compact 0..k-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _local_gain(A, labels, i, m):
    """Change in modularity from i's current community assignment.

    A compact proxy: the within-community edge weight of i minus its expected
    weight -- the term of Q that i contributes.
    """
    lab = labels[i]
    members = labels == lab
    k_i = A[i].sum()
    ki_in = A[i][members].sum()
    k_tot = A[members].sum()
    return ki_in / m - k_i * k_tot / (2 * m ** 2)


def label_propagation(A, max_iter=100, random_state=None):
    """Let each node take the majority label of its neighbours, until stable.

    THE IDEA
    --------
    Start with every node uniquely labelled. Repeatedly, in random order, set each
    node's label to whichever label is most common among its neighbours. Densely
    connected groups quickly converge to a shared label, and the labels stop
    changing when the graph has settled into communities. It uses NO objective
    function and NO parameters -- communities emerge purely from local majority
    voting.

    Near-linear time and dead simple, which makes it a favourite for huge graphs.
    The cost of that simplicity: it is stochastic (the random visiting order
    changes the result) and can be unstable -- on symmetric graphs labels can
    oscillate or collapse everything into one community. Run it a few times and
    take the best by modularity.

    Raghavan, Albert & Kumara (2007).
    """
    A = np.asarray(A, float)
    rng = check_random_state(random_state)
    n = len(A)
    labels = np.arange(n)
    adj = [np.where(A[i] > 0)[0] for i in range(n)]

    for _ in range(max_iter):
        changed = False
        for i in rng.permutation(n):
            if len(adj[i]) == 0:
                continue
            # most frequent label among neighbours (ties broken randomly)
            neigh_labels = labels[adj[i]]
            vals, counts = np.unique(neigh_labels, return_counts=True)
            winners = vals[counts == counts.max()]
            new = rng.choice(winners)
            if new != labels[i]:
                labels[i] = new
                changed = True
        if not changed:
            break
    _, labels = np.unique(labels, return_inverse=True)
    return labels


__all__ = ["louvain", "label_propagation", "modularity"]
