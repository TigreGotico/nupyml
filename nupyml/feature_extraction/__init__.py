"""Turning text into vectors a model can consume.

THE BAG OF WORDS
----------------
Represent a document by which words it contains and how often, discarding order
entirely. "Dog bites man" and "man bites dog" become identical -- an obviously
lossy model that works remarkably well, because for most classification tasks
vocabulary carries the signal and syntax is a refinement.

The matrix is enormous (a column per vocabulary word) and almost entirely zeros,
so it is stored sparse. That sparsity is why ``MultinomialNB`` and ``LinearSVC``
handle text so comfortably: cost scales with the words actually present, not with
the vocabulary.

WHY TF-IDF, NOT COUNTS
----------------------
Raw counts are dominated by words that are common everywhere -- "the" appears in
every document and distinguishes nothing. TF-IDF weights each count by how RARE
the word is across the corpus::

    tfidf = tf * log(n_documents / n_documents_containing_word)

A word in every document gets ``log(1) = 0`` and vanishes. A word in a handful
gets a large weight. So the representation emphasises exactly the terms that
discriminate, and the stopword list mostly stops being necessary.

Rows are then L2-normalised, so a long document and a short one on the same
topic land in the same direction -- which is what makes cosine similarity
meaningful, and why ``Normalizer``-style row scaling is the default here.

``ngram_range`` buys back a little of the word order that the bag of words threw
away, at the cost of a much larger vocabulary.
"""
import re

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin, check_is_fitted

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class CountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, ngram_range=(1, 1), min_df=1,
                 max_df=1.0, max_features=None, binary=False, stop_words=None):
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.binary = binary
        self.stop_words = stop_words

    def _tokenize(self, doc):
        if self.lowercase:
            doc = doc.lower()
        tokens = _TOKEN_RE.findall(doc)
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        lo, hi = self.ngram_range
        out = []
        for n in range(lo, hi + 1):
            out.extend(" ".join(tokens[i:i + n])
                       for i in range(len(tokens) - n + 1))
        return out

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        docs = [self._tokenize(d) for d in raw_documents]
        n_docs = len(docs)
        df = {}
        tf = {}
        for tokens in docs:
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
        max_df_count = self.max_df * n_docs if isinstance(self.max_df, float) \
            else self.max_df
        min_df_count = self.min_df * n_docs if isinstance(self.min_df, float) \
            else self.min_df
        terms = [t for t, c in df.items()
                 if min_df_count <= c <= max_df_count]
        if self.max_features is not None:
            terms.sort(key=lambda t: (-tf[t], t))
            terms = terms[: self.max_features]
        terms.sort()
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        return self._count(docs)

    def _count(self, docs):
        vocab = self.vocabulary_
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(docs):
            counts = {}
            for t in tokens:
                j = vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        X = sp.csr_matrix((vals, (rows, cols)),
                          shape=(len(docs), len(vocab)), dtype=np.float64)
        if self.binary:
            X.data[:] = 1.0
        return X

    def transform(self, raw_documents):
        check_is_fitted(self, "vocabulary_")
        return self._count([self._tokenize(d) for d in raw_documents])

    def get_feature_names_out(self):
        check_is_fitted(self, "vocabulary_")
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get))


class TfidfVectorizer(CountVectorizer):
    def __init__(self, lowercase=True, ngram_range=(1, 1), min_df=1,
                 max_df=1.0, max_features=None, stop_words=None,
                 norm="l2", smooth_idf=True, sublinear_tf=False):
        super().__init__(lowercase=lowercase, ngram_range=ngram_range,
                         min_df=min_df, max_df=max_df,
                         max_features=max_features, stop_words=stop_words)
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        n = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        smooth = int(self.smooth_idf)
        self.idf_ = np.log((n + smooth) / (df + smooth)) + 1.0
        return self._tfidf(X)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def transform(self, raw_documents):
        check_is_fitted(self, "idf_")
        return self._tfidf(super().transform(raw_documents))

    def _tfidf(self, X):
        X = X.copy()
        if self.sublinear_tf:
            X.data = 1.0 + np.log(X.data)
        X = X.multiply(self.idf_).tocsr()
        if self.norm == "l2":
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
            norms[norms == 0.0] = 1.0
            X = sp.diags(1.0 / norms) @ X
        elif self.norm == "l1":
            norms = np.abs(X).sum(axis=1).A.ravel()
            norms[norms == 0.0] = 1.0
            X = sp.diags(1.0 / norms) @ X
        return X.tocsr()


from ._hashing import FeatureHasher, DictVectorizer, HashingVectorizer  # noqa: E402

__all__ = ["CountVectorizer", "TfidfVectorizer", "FeatureHasher",
           "DictVectorizer", "HashingVectorizer"]
