"""Graph kernels: a similarity between whole GRAPHS, for feeding graphs to an SVM.

The methods elsewhere here work on ONE graph's nodes. A graph kernel instead
compares two ENTIRE graphs -- two molecules, two social networks -- producing a
similarity that an ordinary kernel method (SVM, kernel ridge) can consume. The
challenge is that graphs have no fixed size or node ordering, so the kernel must
compare STRUCTURE, invariant to how the nodes happen to be numbered.
"""
import numpy as np


def _wl_labels(A, labels, n_iter):
    """Run Weisfeiler-Lehman relabelling and collect the multiset of all labels
    seen across the iterations."""
    A = np.asarray(A)
    n = len(A)
    adj = [np.where(A[i] > 0)[0] for i in range(n)]
    current = list(labels)
    all_labels = list(current)
    for _ in range(n_iter):
        new = []
        for i in range(n):
            # a node's new label = its own label + the SORTED labels of its
            # neighbours. Sorting is what makes it invariant to node numbering
            neigh = sorted(current[j] for j in adj[i])
            new.append(str(current[i]) + "|" + ",".join(map(str, neigh)))
        # compress the long strings to small integer ids (shared across graphs
        # via the caller's vocabulary)
        current = new
        all_labels.extend(new)
    return all_labels


def weisfeiler_lehman_kernel(graphs, node_labels=None, n_iter=3):
    """The Weisfeiler-Lehman subtree kernel: count shared neighbourhood patterns.

    THE IDEA
    --------
    Borrowed from the WL graph-isomorphism test. Each node starts with a label
    (its degree, if none given). In a round, every node's label is REPLACED by a
    hash of its own label plus the SORTED multiset of its neighbours' labels --
    so after ``k`` rounds a node's label encodes its entire ``k``-hop
    neighbourhood as a subtree pattern. Do this for a few rounds, count how often
    each distinct label appears in each graph, and the kernel between two graphs
    is the DOT PRODUCT of those count vectors: graphs that share many
    neighbourhood patterns are similar.

    Why it is the workhorse graph kernel: sorting the neighbour labels makes it
    invariant to node numbering (the whole point), the relabelling is
    near-linear, and it captures rich local structure -- enough that WL-kernel
    SVMs were long the state of the art for molecule and network classification,
    and the WL test is the same expressiveness bound that modern graph neural
    networks are measured against.

    ``graphs`` is a list of adjacency matrices. Returns the (n_graphs, n_graphs)
    kernel (Gram) matrix.

    Shervashidze et al. (2011).
    """
    graphs = [np.asarray(g) for g in graphs]
    if node_labels is None:
        # default node label = degree, a permutation-invariant starting point
        node_labels = [g.sum(axis=1).astype(int).tolist() for g in graphs]

    # collect every graph's label multiset over all WL iterations, into one
    # shared vocabulary so counts are comparable across graphs
    per_graph_labels = [_wl_labels(g, lab, n_iter)
                        for g, lab in zip(graphs, node_labels)]
    vocab = {}
    for labels in per_graph_labels:
        for lab in labels:
            if lab not in vocab:
                vocab[lab] = len(vocab)

    # feature vector per graph = counts of each label pattern
    features = np.zeros((len(graphs), len(vocab)))
    for gi, labels in enumerate(per_graph_labels):
        for lab in labels:
            features[gi, vocab[lab]] += 1

    # the kernel is the dot product of the pattern-count vectors
    return features @ features.T


__all__ = ["weisfeiler_lehman_kernel"]
