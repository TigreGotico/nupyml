"""Supervised (label-guided) embeddings for categorical data.

Most embedding methods embed TEXT. These embed categorical FEATURE DICTS --
``{"color": "red", "size": "small"}`` -- or high-cardinality categorical columns,
by training a CLASSIFIER and reading representations out of it. The labels GUIDE
what clusters together: rows that predict the same class end up near each other.
Because nupyml's autograd trains it and numpy runs it, there is no torch/ONNX
split -- inference is plain numpy.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, ClassifierMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from .. import nn
from ..autograd import Tensor


class CategoricalVectorizer(BaseEstimator, TransformerMixin):
    """One-hot encode a list of categorical feature DICTS.

    Learns the set of keys and, per key, the set of values seen in training. At
    transform time each dict becomes the concatenation of per-key one-hot blocks;
    an unseen value (or missing key) yields an all-zero block for that key, so the
    output width is fixed and test-time novelty degrades gracefully.
    """

    def fit(self, X, y=None):
        keys = set()
        for d in X:
            keys.update(d.keys())
        self.keys_ = sorted(keys)
        self.categories_ = {}
        for k in self.keys_:
            vals = sorted({str(d[k]) for d in X if k in d})
            self.categories_[k] = {v: i for i, v in enumerate(vals)}
        self.offsets_, off = {}, 0
        for k in self.keys_:
            self.offsets_[k] = off
            off += len(self.categories_[k])
        self.n_features_ = off
        return self

    def transform(self, X):
        check_is_fitted(self, "keys_")
        out = np.zeros((len(X), self.n_features_))
        for i, d in enumerate(X):
            for k in self.keys_:
                if k in d:
                    idx = self.categories_[k].get(str(d[k]))
                    if idx is not None:               # unseen value -> all-zero block
                        out[i, self.offsets_[k] + idx] = 1.0
        return out


class LabelGuidedEmbeddings(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Embed categorical dicts via a label-trained MLP's hidden layer.

    THE METHOD
    ----------
    Vectorise the dicts (one-hot), train an MLP to CLASSIFY the labels, and take
    the activations of the last hidden layer as the embedding. Since the network
    only keeps what helps predict the label, the embedding space organises itself
    by the supervised signal -- items that behave alike for the task cluster
    together, and the geometry is meaningful (nearby = similar for the task) in a
    way an unsupervised or target-mean encoding is not.

    ``transform`` returns the ``embedding_size``-dim activations; ``predict``
    returns the class. Train with the autograd stack, run inference in numpy.
    """

    def __init__(self, hidden=(32,), embedding_size=8, epochs=150, lr=0.01,
                 batch_size=32, random_state=None):
        self.hidden = hidden
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        self.vectorizer_ = CategoricalVectorizer().fit(X)
        Xoh = self.vectorizer_.transform(X)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)

        dims = [Xoh.shape[1]] + list(self.hidden) + [self.embedding_size]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.encoder_ = nn.Sequential(*layers)       # ends at the embedding, ReLU'd
        self.head_ = nn.Linear(self.embedding_size, len(self.classes_))

        params = list(self.encoder_.parameters()) + list(self.head_.parameters())
        opt = nn.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(Xoh)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                emb = self.encoder_(Tensor(Xoh[b]))
                logits = self.head_(emb)
                loss_fn(logits, yi[b]).backward()
                opt.step()
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")
        Xoh = self.vectorizer_.transform(X)
        return self.encoder_(Tensor(Xoh)).data

    def _logits(self, X):
        Xoh = self.vectorizer_.transform(X)
        return self.head_(self.encoder_(Tensor(Xoh))).data

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def predict_proba(self, X):
        z = self._logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class EntityEmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Learned dense vectors for high-cardinality categorical COLUMNS.

    THE KAGGLE-WINNING TRICK
    ------------------------
    One-hot encoding a 10000-value category makes a 10000-wide sparse block that
    treats every value as equidistant. Entity embeddings instead learn a short
    dense vector per value -- trained through a supervised task -- so that similar
    values (nearby ZIP codes, related products) end up with similar vectors, and
    the model sees a compact, geometry-carrying representation. It is the tabular
    analogue of word embeddings, and it complements the target-statistic
    ``encoders`` with a LEARNED alternative.

    Each column gets its own embedding table (size ``embedding_dim``); the tables
    are trained jointly through an MLP classifier, and ``transform`` returns the
    concatenated per-column embeddings.
    """

    def __init__(self, embedding_dim=4, hidden=(32,), epochs=150, lr=0.01,
                 batch_size=32, random_state=None):
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=object)
        self._encoders = [LabelEncoder().fit(X[:, j]) for j in range(X.shape[1])]
        codes = np.column_stack([e.transform(X[:, j])
                                 for j, e in enumerate(self._encoders)])
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)

        self.tables_ = [nn.Embedding(len(e.classes_), self.embedding_dim)
                        for e in self._encoders]
        cat_dim = self.embedding_dim * X.shape[1]
        dims = [cat_dim] + list(self.hidden) + [len(self.classes_)]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.mlp_ = nn.Sequential(*layers[:-1])      # drop the trailing ReLU

        params = [p for t in self.tables_ for p in t.parameters()]
        params += list(self.mlp_.parameters())
        opt = nn.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(codes)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                emb = self._embed(codes[b])
                loss_fn(self.mlp_(emb), yi[b]).backward()
                opt.step()
        return self

    def _embed(self, codes):
        parts = [self.tables_[j](codes[:, j].astype(int))
                 for j in range(codes.shape[1])]
        return Tensor.concatenate(parts, axis=1)

    def transform(self, X):
        check_is_fitted(self, "tables_")
        X = np.asarray(X, dtype=object)
        codes = np.column_stack([e.transform(X[:, j])
                                 for j, e in enumerate(self._encoders)])
        return self._embed(codes).data


__all__ = ["CategoricalVectorizer", "LabelGuidedEmbeddings",
           "EntityEmbeddingEncoder"]
