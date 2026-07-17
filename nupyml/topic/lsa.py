"""Latent Semantic Analysis: topics as the singular vectors of the term matrix.

THE IDEA
--------
Build the term-document matrix (rows = words, columns = documents, entries =
tf-idf weights) and take its truncated SVD. The top ``k`` left singular vectors
are the "topics" -- directions in word space along which documents vary most --
and each document's coordinates in that ``k``-dimensional space are its topic
mixture. No probability model, just the observation that a good low-rank
approximation of the term-document matrix captures the dominant co-occurrence
patterns, which read as themes.

WHY IT WORKS ON SYNONYMY
------------------------
Two documents that share no exact words but use SYNONYMS ("car" vs "automobile")
look unrelated in raw word space, but if those synonyms co-occur with the same
other words across the corpus, the SVD folds them onto the same topic direction.
So LSA matches documents on MEANING rather than surface words -- the original
motivation (Latent Semantic Indexing, for search) and still its clearest payoff.

It is older and cruder than LDA (no sparsity, no probabilities, and the topic
axes can have negative, hard-to-interpret entries), but it is a single SVD --
fast, deterministic, and hyperparameter-free beyond the number of topics.

Deerwester, Dumais, Furnas, Landauer & Harshman (1990).
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted


class LatentSemanticAnalysis(BaseEstimator, TransformerMixin):
    """Topic extraction by truncated SVD of a tf-idf matrix.

    ``fit`` takes a list of raw document strings, vectorises them with tf-idf, and
    keeps the top ``n_topics`` singular directions. ``transform`` projects new
    documents into that topic space; ``top_words`` reads a topic off its singular
    vector.
    """

    def __init__(self, n_topics=10):
        self.n_topics = n_topics

    def fit(self, documents, y=None):
        from ..feature_extraction import TfidfVectorizer
        self._vectorizer = TfidfVectorizer()
        # documents x terms, tf-idf weighted (rare-but-present words weigh more)
        X = self._vectorizer.fit_transform(documents)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self._mean = X.mean(axis=0)

        # truncated SVD: the top singular directions are the topics
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.n_topics, len(s))
        self.components_ = Vt[:k]                # (n_topics, n_terms)
        self.singular_values_ = s[:k]
        self.explained_variance_ratio_ = (s[:k] ** 2) / (s ** 2).sum()
        # map term index back to the word, for top_words
        vocab = self._vectorizer.vocabulary_
        self.index_to_word_ = {i: w for w, i in vocab.items()}
        return self

    def transform(self, documents):
        """Project documents onto the topic axes -> their topic coordinates."""
        check_is_fitted(self, "components_")
        X = self._vectorizer.transform(documents)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return X @ self.components_.T

    def top_words(self, topic, n=10):
        """The words with the largest weight on a topic's singular vector."""
        check_is_fitted(self, "components_")
        order = np.argsort(np.abs(self.components_[topic]))[::-1][:n]
        return [self.index_to_word_[i] for i in order]


__all__ = ["LatentSemanticAnalysis"]
