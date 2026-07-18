"""I1: embeddings v2 -- PPMI+SVD, item2vec, FastText, Poincare, StarSpace.

Each is held to its defining property: PPMI vectors are close for co-occurring
words; item2vec neighbours are co-basket items; FastText gives OOV words a
sensible vector from subwords; Poincare embeds a hierarchy so connected nodes are
closer; StarSpace classifies via the shared space.
"""
import numpy as np
import pytest

from nupyml.embed import ppmi_svd, Item2Vec, FastText, PoincareEmbedding, StarSpace


def _cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def test_ppmi_svd_captures_cooccurrence():
    sents = ([["king", "queen", "royal"]] * 20 + [["car", "road", "drive"]] * 20
             + [["king", "royal", "crown"]] * 10)
    vocab, vec = ppmi_svd(sents, dim=5, window=2)
    assert _cos(vec["king"], vec["royal"]) > _cos(vec["king"], vec["car"])


def test_item2vec_neighbours_are_co_basket():
    baskets = [["a", "b", "c"]] * 30 + [["x", "y", "z"]] * 30 + [["a", "b"]] * 10
    i2v = Item2Vec(dim=16, epochs=30, random_state=0).fit(baskets)
    sim = i2v.most_similar("a", k=2)
    assert set(sim) <= {"b", "c"}                      # a's neighbours are its basket-mates


def test_fasttext_gives_oov_words_a_vector():
    sents = [["running", "runner", "run", "walking", "walk", "walker"]] * 25
    ft = FastText(dim=16, epochs=15, random_state=0).fit(sents)
    v_run = ft.get_vector("run")
    v_runs = ft.get_vector("runs")                     # OOV: never seen in training
    assert v_runs.shape == (16,)
    # the OOV word shares subwords with 'run' -> positive similarity
    assert _cos(v_run, v_runs) > 0.2


def test_poincare_embeds_hierarchy():
    edges = [("root", "a"), ("root", "b"), ("root", "c"),
             ("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2"), ("c", "c1")]
    pc = PoincareEmbedding(dim=2, epochs=300, lr=0.5, random_state=0).fit(edges)
    # connected pair closer than an unconnected cross-branch pair
    assert pc.distance("a", "a1") < pc.distance("a", "b1")
    # all points stay strictly inside the Poincare ball
    assert all(pc.norm(n) < 1.0 for n in pc.nodes_)


def test_starspace_classifies_via_shared_space():
    bags = [[0, 1, 2]] * 30 + [[3, 4, 5]] * 30
    labels = ["p"] * 30 + ["q"] * 30
    ss = StarSpace(dim=16, epochs=30, random_state=0).fit(bags, labels)
    assert (ss.predict(bags) == np.array(labels)).mean() > 0.9
    # a bag of class-p features is classified p
    assert ss.predict([[0, 1]])[0] == "p"
