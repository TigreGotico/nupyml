"""F7: link prediction, Girvan-Newman, assortativity, max-flow/min-cut, kernels, LINE.

Link predictors must rank truly-missing edges above random non-edges;
Girvan-Newman must recover planted communities; assortativity must separate an
assortative graph from a star; max-flow must equal min-cut (the theorem); the LINE
embedding must place connected nodes closer than unconnected ones.
"""
import numpy as np
import pytest

from nupyml.graph import (common_neighbors, jaccard_coefficient,
                         adamic_adar_index, resource_allocation_index,
                         preferential_attachment, katz_index,
                         degree_assortativity, girvan_newman, max_flow, min_cut,
                         shortest_path_kernel, random_walk_kernel, LINE)
from nupyml.metrics import adjusted_rand_score


def _two_community_graph(seed=0, n=20, p_in=0.5, p_out=0.05):
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n // 2) == (j < n // 2)
            if rng.rand() < (p_in if same else p_out):
                A[i, j] = A[j, i] = 1
    true = np.array([0] * (n // 2) + [1] * (n // 2))
    return A, true


# --- link prediction ------------------------------------------------------

@pytest.mark.parametrize("scorer", [
    common_neighbors, jaccard_coefficient, adamic_adar_index,
    resource_allocation_index, katz_index,
], ids=["cn", "jaccard", "adamic_adar", "resource_alloc", "katz"])
def test_link_predictor_ranks_missing_edges_above_nonedges(scorer):
    rng = np.random.RandomState(0)
    A, _ = _two_community_graph(seed=1, n=30, p_in=0.6)
    # hide 20 existing edges
    edges = np.transpose(np.nonzero(np.triu(A, 1)))
    hidden = edges[rng.choice(len(edges), 20, replace=False)]
    Atr = A.copy()
    for i, j in hidden:
        Atr[i, j] = Atr[j, i] = 0
    S = scorer(Atr)
    hidden_scores = [S[i, j] for i, j in hidden]
    # sample non-edges (never connected in the full graph)
    nonedges = np.transpose(np.nonzero((A == 0) & (np.triu(np.ones_like(A), 1) > 0)))
    ne = nonedges[rng.choice(len(nonedges), 200, replace=False)]
    nonedge_scores = [S[i, j] for i, j in ne]
    # AUC: a hidden edge outscores a random non-edge more often than not
    auc = np.mean([h > n for h in hidden_scores for n in nonedge_scores])
    assert auc > 0.6


def test_preferential_attachment_is_degree_product():
    A, _ = _two_community_graph()
    deg = (A > 0).sum(axis=1)
    S = preferential_attachment(A)
    assert S[3, 7] == pytest.approx(deg[3] * deg[7])


# --- communities & assortativity ------------------------------------------

def test_girvan_newman_recovers_communities():
    A, true = _two_community_graph(p_in=0.6, p_out=0.03)
    labels = girvan_newman(A, n_communities=2)
    assert adjusted_rand_score(true, labels) > 0.7


def test_assortativity_sign():
    # a star: one hub linked to many leaves -> disassortative (negative)
    n = 12
    star = np.zeros((n, n))
    star[0, 1:] = star[1:, 0] = 1
    assert degree_assortativity(star) < 0
    # a ring lattice where all degrees are equal -> ~0 / undefined-ish, but a
    # graph pairing similar degrees is assortative
    A = np.zeros((6, 6))
    # two triangles (all degree 2) joined by one edge: mostly equal degrees
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        A[a, b] = A[b, a] = 1
    assert degree_assortativity(A) > degree_assortativity(star)


# --- flow ------------------------------------------------------------------

def test_max_flow_equals_min_cut():
    cap = np.array([[0, 3, 2, 0],
                    [0, 0, 1, 3],
                    [0, 0, 0, 2],
                    [0, 0, 0, 0]], float)
    flow, _ = max_flow(cap, 0, 3)
    cut_value, source_side = min_cut(cap, 0, 3)
    assert flow == pytest.approx(cut_value)          # max-flow min-cut theorem
    assert flow == pytest.approx(5.0)
    assert 0 in source_side and 3 not in source_side


def test_min_cut_separates_a_bottleneck():
    # two clusters joined by a single low-capacity edge
    cap = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        cap[a, b] = cap[b, a] = 10
    for a, b in [(3, 4), (3, 5), (4, 5)]:
        cap[a, b] = cap[b, a] = 10
    cap[2, 3] = cap[3, 2] = 1                          # the bottleneck
    value, side = min_cut(cap, 0, 5)
    assert value == pytest.approx(1.0)                 # cut the single weak edge
    assert set(side.tolist()) == {0, 1, 2}


# --- kernels ---------------------------------------------------------------

def test_shortest_path_kernel_is_permutation_invariant():
    A, _ = _two_community_graph(seed=2, n=12)
    perm = np.random.RandomState(0).permutation(12)
    A_perm = A[perm][:, perm]                          # relabel the nodes
    k_self = shortest_path_kernel(A, A)
    k_perm = shortest_path_kernel(A, A_perm)
    assert k_self == pytest.approx(k_perm)             # relabelling changes nothing


def test_random_walk_kernel_higher_for_similar_graphs():
    A, _ = _two_community_graph(seed=3, n=10)
    empty = np.zeros((10, 10))
    assert random_walk_kernel(A, A) > random_walk_kernel(A, empty)


# --- LINE ------------------------------------------------------------------

def test_line_embeds_connected_nodes_closer():
    A, true = _two_community_graph(seed=0, n=24, p_in=0.6, p_out=0.02)
    emb = LINE(n_components=16, n_epochs=60, random_state=0).fit_transform(A)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = emb @ emb.T
    edges = np.triu(A, 1) > 0
    nonedges = (A == 0) & (np.triu(np.ones_like(A), 1) > 0)
    assert sim[edges].mean() > sim[nonedges].mean()
