"""Graph algorithms: importance, community detection, embeddings, kernels.

Most tests build a two-community graph (dense within, sparse between) and check
each method recovers the structure it should: importance ranks the hub, community
detection recovers the two groups, embeddings cluster by community, and the WL
kernel scores identical graphs above different ones.
"""
import numpy as np
import pytest

from nupyml.metrics import adjusted_rand_score
from nupyml.graph import (pagerank, hits, centrality, louvain,
                         label_propagation, modularity, DeepWalk, Node2Vec,
                         weisfeiler_lehman_kernel)


@pytest.fixture
def two_communities():
    rng = np.random.RandomState(0)
    n = 24
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < 12) == (j < 12)
            if rng.uniform() < (0.6 if same else 0.04):
                A[i, j] = A[j, i] = 1
    truth = np.array([0] * 12 + [1] * 12)
    return A, truth


# --- importance -----------------------------------------------------------

def test_pagerank_is_a_distribution(two_communities):
    A, _ = two_communities
    pr = pagerank(A)
    assert pr.sum() == pytest.approx(1.0)
    assert np.all(pr >= 0)


def test_pagerank_ranks_a_hub_highest():
    """A star: the centre should get the most PageRank."""
    n = 10
    A = np.zeros((n, n))
    A[0, 1:] = 1                                # node 0 points to all others
    A[1:, 0] = 1
    pr = pagerank(A)
    assert np.argmax(pr) == 0


def test_pagerank_handles_dangling_nodes():
    """A node with no out-links must not absorb all the rank (the teleport fix)."""
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)   # node 2 dangles
    pr = pagerank(A)
    assert np.all(np.isfinite(pr))
    assert pr.sum() == pytest.approx(1.0)


def test_hits_returns_hub_and_authority_scores(two_communities):
    A, _ = two_communities
    hubs, auth = hits(A)
    assert hubs.shape == (len(A),) and auth.shape == (len(A),)


def test_centralities_disagree_meaningfully():
    """A bridge node joining two clusters has high betweenness but low degree --
    the measures capture different notions of 'central'."""
    # two triangles joined by a single bridge node (index 3)
    A = np.zeros((7, 7))
    for i, j in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 6)]:
        A[i, j] = A[j, i] = 1
    deg = centrality(A, "degree")
    btw = centrality(A, "betweenness")
    # node 3 (the bridge) has modest degree but the highest betweenness
    assert np.argmax(btw) == 3
    assert np.argmax(deg) != 3


def test_eigenvector_centrality_is_a_distribution(two_communities):
    A, _ = two_communities
    c = centrality(A, "eigenvector")
    assert c.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(c >= 0)


# --- community detection --------------------------------------------------

def test_louvain_recovers_communities(two_communities):
    A, truth = two_communities
    labels = louvain(A, random_state=0)
    assert adjusted_rand_score(truth, labels) > 0.8


def test_louvain_discovers_the_number_of_communities(two_communities):
    """Unlike k-means, the count is found, not specified -- and here it should be
    the true two."""
    A, _ = two_communities
    labels = louvain(A, random_state=0)
    assert len(set(labels)) == 2


def test_label_propagation_recovers_communities(two_communities):
    A, truth = two_communities
    labels = label_propagation(A, random_state=0)
    assert adjusted_rand_score(truth, labels) > 0.8


def test_modularity_prefers_the_true_partition(two_communities):
    """The true community split should score higher modularity than a random one."""
    A, truth = two_communities
    rng = np.random.RandomState(0)
    random_partition = rng.randint(0, 2, len(A))
    assert modularity(A, truth) > modularity(A, random_partition)


def test_modularity_of_no_edges_is_zero():
    assert modularity(np.zeros((5, 5)), np.arange(5)) == 0.0


# --- node embeddings ------------------------------------------------------

def test_deepwalk_clusters_by_community(two_communities):
    """Nodes in the same community should have more similar embeddings than nodes
    across communities."""
    A, _ = two_communities
    dw = DeepWalk(n_dim=32, walk_length=20, n_walks=10, n_epochs=5,
                  random_state=0).fit(A)
    same = np.mean([dw.similarity(i, j) for i in range(12) for j in range(i + 1, 12)])
    cross = np.mean([dw.similarity(i, j) for i in range(12) for j in range(12, 24)])
    assert same > cross


def test_deepwalk_embedding_shape(two_communities):
    A, _ = two_communities
    dw = DeepWalk(n_dim=16, n_walks=5, walk_length=10, n_epochs=3,
                  random_state=0).fit(A)
    assert dw.embeddings_.shape == (len(A), 16)


def test_node2vec_biased_walks_run(two_communities):
    """node2vec with p/q != 1 uses biased walks; it should still embed and
    cluster."""
    A, _ = two_communities
    nv = Node2Vec(n_dim=32, walk_length=20, n_walks=10, n_epochs=5,
                  p=0.5, q=2.0, random_state=0).fit(A)
    same = np.mean([nv.similarity(i, j) for i in range(12) for j in range(i + 1, 12)])
    cross = np.mean([nv.similarity(i, j) for i in range(12) for j in range(12, 24)])
    assert same > cross


# --- graph kernel ---------------------------------------------------------

def test_wl_kernel_scores_identical_graphs_highest():
    """Two copies of a graph must be more similar than either is to a different
    graph -- and a graph is most similar to itself."""
    rng = np.random.RandomState(0)
    g1 = (rng.uniform(size=(10, 10)) < 0.3).astype(float)
    g1 = np.triu(g1, 1); g1 = g1 + g1.T
    g2 = g1.copy()
    g3 = np.zeros((10, 10)); g3[:4, :4] = 1 - np.eye(4)     # a small clique
    K = weisfeiler_lehman_kernel([g1, g2, g3])
    assert K[0, 1] > K[0, 2]                    # identical > different
    assert K[0, 0] >= K[0, 2]


def test_wl_kernel_is_symmetric_and_positive():
    rng = np.random.RandomState(1)
    graphs = []
    for _ in range(4):
        g = (rng.uniform(size=(8, 8)) < 0.4).astype(float)
        g = np.triu(g, 1); graphs.append(g + g.T)
    K = weisfeiler_lehman_kernel(graphs)
    assert np.allclose(K, K.T)                  # a kernel matrix is symmetric
    assert np.all(K >= 0)                       # count dot-products are nonnegative


def test_wl_kernel_is_permutation_invariant():
    """Relabelling a graph's nodes must not change its similarity to another --
    the invariance the sorted-neighbour trick provides."""
    rng = np.random.RandomState(0)
    g = (rng.uniform(size=(8, 8)) < 0.4).astype(float)
    g = np.triu(g, 1); g = g + g.T
    perm = rng.permutation(8)
    g_permuted = g[np.ix_(perm, perm)]          # same graph, renumbered
    K = weisfeiler_lehman_kernel([g, g_permuted])
    assert K[0, 1] == pytest.approx(K[0, 0])    # identical up to relabelling
