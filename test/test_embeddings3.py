"""J6: embeddings v3 -- LSH-ANN, feature hashing, struc2vec, metapath2vec.

LSH must retrieve most true nearest neighbours while scanning far fewer points;
feature hashing must map to a fixed size deterministically; struc2vec must place
structural TWINS closer than a hub and a leaf; metapath2vec must place co-authors
closer than non-co-authors.
"""
import numpy as np
import pytest

from nupyml.embed import LSHIndex, feature_hashing, struc2vec, metapath2vec


def _cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def test_lsh_retrieves_neighbours_scanning_fewer_points():
    rng = np.random.RandomState(0)
    X = rng.randn(1000, 20)
    q = X[0]
    lsh = LSHIndex(n_bits=10, n_tables=20, random_state=0).fit(X)
    approx = set(lsh.query(q, k=10))
    sims = X @ q / (np.linalg.norm(X, axis=1) * np.linalg.norm(q))
    true = set(np.argsort(sims)[::-1][:10])
    assert len(approx & true) / 10 >= 0.5                 # decent recall
    assert len(lsh._candidates(q)) < 200                  # scans << 1000 points


def test_feature_hashing_fixed_size_and_deterministic():
    recs = [{'a': 1, 'b': 2}, ['x', 'y', 'x'], {'z': 3}]
    H = feature_hashing(recs, n_features=16)
    assert H.shape == (3, 16)
    assert np.allclose(H, feature_hashing(recs, n_features=16))   # deterministic
    # the token list 'x','y','x' put weight in exactly (at most) two columns
    assert np.count_nonzero(H[1]) <= 2


def test_struc2vec_embeds_structural_role():
    # two disjoint stars: hubs (deg 4) are structural twins; leaves (deg 1) too
    A = np.zeros((10, 10))
    for leaf in (1, 2, 3, 4):
        A[0, leaf] = A[leaf, 0] = 1
    for leaf in (6, 7, 8, 9):
        A[5, leaf] = A[leaf, 5] = 1
    s = struc2vec(n_dim=16, n_walks=40, walk_length=15, n_hops=2, epochs=15,
                  random_state=0).fit(A)
    # the two hubs (same role, different component) are closer than a hub and a leaf
    assert _cos(s.get_vector(0), s.get_vector(5)) > _cos(s.get_vector(0),
                                                         s.get_vector(1))


def test_metapath2vec_places_coauthors_together():
    # types: 0=author, 1=paper. Authors 0,1 share papers 3,4; author 2 is separate.
    A = np.zeros((6, 6))
    types = [0, 0, 0, 1, 1, 1]
    for a, p in [(0, 3), (1, 3), (0, 4), (1, 4), (2, 5)]:
        A[a, p] = A[p, a] = 1
    mp = metapath2vec(metapath=[0, 1, 0], n_dim=16, n_walks=60, walk_length=20,
                      epochs=15, random_state=0).fit(A, types)
    assert _cos(mp.get_vector(0), mp.get_vector(1)) > _cos(mp.get_vector(0),
                                                           mp.get_vector(2))
