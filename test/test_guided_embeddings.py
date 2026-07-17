"""G4: supervised (label-guided) categorical embeddings.

The vectorizer must one-hot dicts and degrade gracefully on unseen values; the
label-guided embedding must classify well AND arrange its space by label (same-
class points closer than different-class); entity embeddings must produce usable
dense features for high-cardinality columns.
"""
import numpy as np
import pytest

from scipy.spatial.distance import cdist

from nupyml.embed import (CategoricalVectorizer, LabelGuidedEmbeddings,
                         EntityEmbeddingEncoder)


# --- vectorizer -----------------------------------------------------------

def test_categorical_vectorizer_one_hot_and_unseen():
    X = [{"color": "red", "size": "s"},
         {"color": "green", "size": "l"},
         {"color": "red", "size": "m"}]
    cv = CategoricalVectorizer().fit(X)
    T = cv.transform(X)
    assert T.shape == (3, cv.n_features_)
    assert T[0].sum() == 2                            # one bit per key (2 keys)
    # unseen colour -> zero block for colour, seen size -> its bit
    unseen = cv.transform([{"color": "blue", "size": "s"}])
    assert unseen.sum() == 1


# --- label-guided embeddings ----------------------------------------------

def _fruit_data(seed=0, n=150):
    rng = np.random.RandomState(seed)
    fruits = {"apple": ("red", "medium", "round"),
              "cucumber": ("green", "large", "oblong"),
              "banana": ("yellow", "medium", "curved")}
    names = list(fruits)
    X, y = [], []
    for _ in range(n):
        f = names[rng.randint(3)]
        c, s, sh = fruits[f]
        if rng.rand() < 0.2:                          # 20% noisy colour
            c = ("red", "green", "yellow")[rng.randint(3)]
        X.append({"color": c, "size": s, "shape": sh})
        y.append(f)
    return X, np.array(y)


def test_label_guided_embeddings_classify_and_cluster():
    X, y = _fruit_data()
    lge = LabelGuidedEmbeddings(hidden=(16,), embedding_size=4, epochs=150,
                                random_state=0).fit(X, y)
    emb = lge.transform(X)
    assert emb.shape == (len(X), 4)
    assert (lge.predict(X) == y).mean() > 0.9         # the guiding classifier works
    # embeddings organise by label: same-class pairs are closer than cross-class
    D = cdist(emb, emb)
    same = D[(y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool)].mean()
    diff = D[y[:, None] != y[None, :]].mean()
    assert same < diff


def test_label_guided_predict_proba_normalised():
    X, y = _fruit_data()
    lge = LabelGuidedEmbeddings(embedding_size=4, epochs=60, random_state=0).fit(X, y)
    p = lge.predict_proba(X[:10])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.shape == (10, 3)


def test_label_guided_generalises_to_held_out():
    X, y = _fruit_data(seed=1, n=200)
    Xtr, ytr, Xte, yte = X[:140], y[:140], X[140:], y[140:]
    lge = LabelGuidedEmbeddings(hidden=(16,), embedding_size=4, epochs=150,
                                random_state=0).fit(Xtr, ytr)
    assert (lge.predict(Xte) == yte).mean() > 0.85


# --- entity embeddings ----------------------------------------------------

def test_entity_embeddings_produce_usable_features():
    rng = np.random.RandomState(0)
    cols0 = ["a", "b", "c", "d"]
    cols1 = ["x", "y", "z"]
    X, y = [], []
    for _ in range(300):
        i, j = rng.randint(4), rng.randint(3)
        X.append([cols0[i], cols1[j]])
        y.append(int(i < 2))                          # label driven by col0's value
    X = np.array(X, dtype=object); y = np.array(y)
    enc = EntityEmbeddingEncoder(embedding_dim=3, hidden=(16,), epochs=150,
                                 random_state=0).fit(X, y)
    E = enc.transform(X)
    assert E.shape == (300, 2 * 3)                     # 2 columns x 3 dims
    assert np.all(np.isfinite(E))
    # a linear classifier on the learned embeddings beats chance
    from nupyml.linear_model import LogisticRegression
    acc = (LogisticRegression(max_iter=500).fit(E, y).predict(E) == y).mean()
    assert acc > 0.7
