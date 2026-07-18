"""L11: embeddings v5 / retrieval -- GraRep, HOPE, personalized-PageRank, IVFPQ.

GraRep, HOPE and personalized-PageRank all embed nodes so that same-community nodes
are closer than cross-community ones; the IVFPQ index retrieves the true nearest
neighbours while scanning only a few cells.
"""
import numpy as np
import pytest

from nupyml.embed import (GraRep, HOPE, PersonalizedPageRankEmbedding, IVFPQIndex)


def _cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def _sbm(seed=0):
    rng = np.random.RandomState(seed)
    n = 30
    lab = np.array([0] * 15 + [1] * 15)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.5 if lab[i] == lab[j] else 0.05
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    return A


def _same_beats_diff(model):
    A = _sbm()
    m = model.fit(A)
    same = np.mean([_cos(m.get_vector(i), m.get_vector(j))
                    for i in range(15) for j in range(i + 1, 15)])
    diff = np.mean([_cos(m.get_vector(i), m.get_vector(j))
                    for i in range(15) for j in range(15, 30)])
    return same > diff


def test_grarep_recovers_communities():
    assert _same_beats_diff(GraRep(dim=12, max_order=3))


def test_hope_recovers_communities():
    assert _same_beats_diff(HOPE(dim=8, beta=0.05))


def test_personalized_pagerank_recovers_communities():
    assert _same_beats_diff(PersonalizedPageRankEmbedding(dim=8))


def test_ivfpq_retrieves_nearest_neighbours():
    rng = np.random.RandomState(0)
    X = rng.randn(800, 16)
    q = X[3]
    idx = IVFPQIndex(n_cells=16, n_probe=4, n_subvectors=4, n_codes=64,
                     random_state=0).fit(X)
    approx = set(idx.query(q, k=10))
    d = np.linalg.norm(X - q, axis=1)
    true = set(np.argsort(d)[:10])
    assert len(approx & true) / 10 >= 0.6
