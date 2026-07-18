"""K10: embeddings v4 / retrieval -- GraphWave, NetMF, hash embeddings, HNSW.

GraphWave places structural twins together; NetMF (the matrix-factorisation view of
node2vec) recovers communities; hash embeddings vectorise any token into a fixed
pool deterministically; HNSW retrieves the true nearest neighbours while scanning a
fraction of the data.
"""
import numpy as np
import pytest

from nupyml.embed import GraphWave, NetMF, HashEmbedding, HNSW


def _cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def test_graphwave_embeds_structural_role():
    A = np.zeros((10, 10))
    for leaf in (1, 2, 3, 4):
        A[0, leaf] = A[leaf, 0] = 1
    for leaf in (6, 7, 8, 9):
        A[5, leaf] = A[leaf, 5] = 1
    gw = GraphWave(scales=(0.5, 1.0), n_samples=8).fit(A)
    # the two star hubs (structural twins) are closer than a hub and a leaf
    assert _cos(gw.get_vector(0), gw.get_vector(5)) > _cos(gw.get_vector(0),
                                                           gw.get_vector(1))


def test_netmf_recovers_communities():
    rng = np.random.RandomState(0)
    n = 30
    lab = np.array([0] * 15 + [1] * 15)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.5 if lab[i] == lab[j] else 0.05
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    nm = NetMF(dim=8, window=3).fit(A)
    same = np.mean([_cos(nm.get_vector(i), nm.get_vector(j))
                    for i in range(15) for j in range(i + 1, 15)])
    diff = np.mean([_cos(nm.get_vector(i), nm.get_vector(j))
                    for i in range(15) for j in range(15, 30)])
    assert same > diff


def test_hash_embedding_handles_any_token():
    he = HashEmbedding(pool_size=64, dim=16, n_hashes=2).fit()
    v1 = he.get_vector('hello')
    assert v1.shape == (16,)
    assert np.allclose(v1, he.get_vector('hello'))       # deterministic
    assert he.get_vector('never_seen_token_123').shape == (16,)   # any token works
    assert not np.allclose(v1, he.get_vector('world'))   # distinct tokens differ


def test_hnsw_retrieves_nearest_neighbours():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 10)
    q = X[7]
    hnsw = HNSW(M=8, ef=32, random_state=0).fit(X)
    approx = set(hnsw.query(q, k=10))
    d = np.linalg.norm(X - q, axis=1)
    true = set(np.argsort(d)[:10])
    assert len(approx & true) / 10 >= 0.6                # high recall
