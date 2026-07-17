"""Feature hashing and dict vectorization: features without a vocabulary.

THE VOCABULARY PROBLEM
----------------------
``CountVectorizer`` and friends must first BUILD a vocabulary -- scan all the
data, assign each distinct token a column index, and store that map. For a
streaming or web-scale problem that map can be enormous and must be kept in memory
and in sync between training and serving. The hashing trick removes it entirely.
"""
import numpy as np
from scipy.sparse import csr_matrix

from ..base import BaseEstimator, TransformerMixin


def _hash(s, seed=0):
    import hashlib
    h = hashlib.blake2b(str(s).encode(), digest_size=8,
                        salt=(seed % 65536).to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


class FeatureHasher(BaseEstimator, TransformerMixin):
    """The hashing trick: map features to columns by HASHING, no vocabulary.

    THE IDEA
    --------
    Instead of learning "feature X -> column 7", just HASH the feature name to a
    column in a fixed-size output: ``column = hash(name) % n_features``. No
    vocabulary is stored, nothing must be fit, and a feature never seen before
    still gets a column -- so training and serving need no shared state at all.
    This is what makes it work on unbounded, streaming feature spaces.

    THE COST: COLLISIONS
    --------------------
    Two different features can hash to the same column and get added together --
    a collision that mixes their signals. With enough columns collisions are rare
    and their effect averages out, so a linear model barely notices; shrink the
    table and accuracy degrades gracefully. The ``alternate_sign`` trick -- flip
    the sign for half the hashes -- makes colliding features tend to CANCEL rather
    than reinforce, so the collisions are unbiased in expectation. That is the
    detail that makes hashing safe rather than merely cheap.

    Weinberger et al. (2009).
    """

    def __init__(self, n_features=1024, alternate_sign=True):
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X, y=None):
        return self                             # nothing to learn -- the point

    def transform(self, raw):
        """``raw`` is a sequence of dicts {feature: value} or iterables of names."""
        raw = list(raw)
        rows, cols, data = [], [], []
        for i, sample in enumerate(raw):
            items = sample.items() if isinstance(sample, dict) \
                else ((f, 1) for f in sample)
            for name, value in items:
                h = _hash(name)
                col = h % self.n_features
                # a second hash bit sets the sign, so collisions cancel on average
                sign = 1 if (not self.alternate_sign or (h >> 1) & 1) else -1
                rows.append(i)
                cols.append(col)
                data.append(sign * value)
        return csr_matrix((data, (rows, cols)), shape=(len(raw), self.n_features))


class DictVectorizer(BaseEstimator, TransformerMixin):
    """Turn dicts of features into a matrix, one-hot encoding string values.

    The vocabulary-BASED counterpart to FeatureHasher, for when the feature space
    is small enough to enumerate. It learns a column per feature: numeric features
    map to a column directly, and a string value ``{"city": "Paris"}`` becomes a
    one-hot column ``"city=Paris"``. So mixed numeric/categorical records become a
    clean numeric matrix with an inspectable, reversible column mapping -- the
    thing you give up with hashing in exchange for that inspectability.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.vocabulary_ = {}
        for sample in X:
            for key, value in sample.items():
                name = f"{key}={value}" if isinstance(value, str) else key
                if name not in self.vocabulary_:
                    self.vocabulary_[name] = len(self.vocabulary_)
        self.feature_names_ = sorted(self.vocabulary_, key=self.vocabulary_.get)
        return self

    def transform(self, X):
        rows = []
        for sample in X:
            row = np.zeros(len(self.vocabulary_))
            for key, value in sample.items():
                name = f"{key}={value}" if isinstance(value, str) else key
                if name in self.vocabulary_:
                    # string -> one-hot (value 1), numeric -> its value
                    row[self.vocabulary_[name]] = 1 if isinstance(value, str) else value
            rows.append(row)
        return np.array(rows)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class HashingVectorizer(BaseEstimator, TransformerMixin):
    """Text vectorization by hashing tokens -- stateless CountVectorizer.

    CountVectorizer's vocabulary is the thing that does not scale: on a huge or
    streaming corpus it grows without bound and must be held in memory. This hashes
    each token straight to a column, so it is stateless and needs no fit -- you can
    vectorize a document stream you have never seen, in constant memory. The trade
    is the same as FeatureHasher's: token collisions, made unbiased by the signed
    hash, and no way to map a column back to its word.
    """

    def __init__(self, n_features=2 ** 18, alternate_sign=True):
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        rows, cols, data = [], [], []
        n = 0
        for i, doc in enumerate(documents):
            n = i + 1
            counts = {}
            for token in doc.lower().split():
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                h = _hash(token)
                sign = 1 if (not self.alternate_sign or (h >> 1) & 1) else -1
                rows.append(i)
                cols.append(h % self.n_features)
                data.append(sign * count)
        return csr_matrix((data, (rows, cols)), shape=(n, self.n_features))


__all__ = ["FeatureHasher", "DictVectorizer", "HashingVectorizer"]
