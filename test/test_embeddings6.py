"""M10: embeddings v6 -- Lorentz hyperbolic, SDNE, Sent2Vec, spherical text.

Lorentz places a parent nearer its child than a leaf in the other branch; SDNE
separates two planted communities; Sent2Vec makes same-topic sentences more similar
than cross-topic; spherical embeddings live on the unit sphere and put co-occurring
words closer than unrelated ones.
"""
import numpy as np
from scipy.spatial.distance import cdist

from nupyml.embed import (LorentzEmbedding, SDNE, Sent2Vec, SphericalTextEmbedding)


def test_lorentz_embeds_hierarchy():
    edges = [("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2"),
             ("b", "b1"), ("b", "b2")]
    le = LorentzEmbedding(dim=2, epochs=200, lr=0.5, random_state=0).fit(edges)
    assert le.distance("a", "a1") < le.distance("a1", "b1")   # parent nearer than cross


def test_sdne_separates_communities():
    rng = np.random.RandomState(0)
    N = 20; A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = 0.7 if (i < 10) == (j < 10) else 0.05
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    emb = SDNE(dim=8, hidden=32, epochs=150, lr=0.02, random_state=0).fit(A).embedding_
    D = cdist(emb, emb)
    intra = (D[:10, :10].mean() + D[10:, 10:].mean()) / 2
    inter = D[:10, 10:].mean()
    assert intra < inter


def test_sent2vec_groups_by_topic():
    sents = [["cat", "sat", "mat"], ["dog", "ran", "park"],
             ["cat", "likes", "mat"], ["dog", "likes", "park"]] * 10
    s2 = Sent2Vec(dim=32, epochs=30, random_state=0).fit(sents)
    cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    v1, v2, v3 = (s2.transform(x) for x in
                  (["cat", "mat"], ["cat", "sat"], ["dog", "park"]))
    assert cos(v1, v2) > cos(v1, v3)


def test_spherical_embeddings_are_on_the_sphere():
    corpus = [["king", "queen", "royal"], ["man", "woman", "person"],
              ["king", "man", "royal"], ["queen", "woman", "royal"]] * 15
    sp = SphericalTextEmbedding(dim=32, epochs=20, window=3, random_state=0).fit(corpus)
    assert np.allclose(np.linalg.norm(sp.W_, axis=1), 1.0, atol=1e-5)   # unit vectors
    assert sp.similarity("king", "royal") > sp.similarity("king", "person")
