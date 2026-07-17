"""Doc2Vec: a dense vector for a whole DOCUMENT, not just a word.

THE EXTENSION OF WORD2VEC
-------------------------
word2vec learns a vector per word by making it predict its neighbours. Doc2Vec
adds one more vector per DOCUMENT and lets it join in the prediction: the
document vector is trained, alongside the word vectors, to help predict the words
that appear in that document. So the document vector comes to summarise "what
this document is about" in the same space as the words -- and two documents can
then be compared, or matched to a query, by cosine similarity.

This is the PV-DBOW variant (distributed bag of words): the document vector alone
predicts its words, via the same skip-gram-with-negative-sampling objective as
word2vec (see ``embed``). It is simpler than PV-DM (which also uses word context)
and works well for document similarity, which is Doc2Vec's main use.

Le & Mikolov (2014).
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class Doc2Vec(BaseEstimator):
    """Learn a vector per document by predicting the words it contains.

    ``fit`` takes a list of tokenised documents. ``document_vectors_`` holds the
    learned per-document embeddings; ``similarity`` compares two by cosine.
    """

    def __init__(self, n_dim=50, negative=5, learning_rate=0.025, n_epochs=20,
                 min_count=1, random_state=None):
        self.n_dim = n_dim
        self.negative = negative
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.min_count = min_count
        self.random_state = random_state

    def fit(self, documents):
        rng = check_random_state(self.random_state)
        # vocabulary and negative-sampling distribution (unigram^0.75, as word2vec)
        counts = {}
        for doc in documents:
            for w in doc:
                counts[w] = counts.get(w, 0) + 1
        self.vocab_ = {w: i for i, w in enumerate(
            w for w, c in counts.items() if c >= self.min_count)}
        V = len(self.vocab_)
        freq = np.array([counts[w] for w in self.vocab_], dtype=float) ** 0.75
        neg_dist = freq / freq.sum()

        n_docs = len(documents)
        # document vectors (trained) and output word vectors (scaffolding)
        self.document_vectors_ = rng.uniform(-0.5, 0.5, (n_docs, self.n_dim)) / self.n_dim
        W_out = np.zeros((V, self.n_dim))
        lr = self.learning_rate

        indexed = [[self.vocab_[w] for w in doc if w in self.vocab_]
                   for doc in documents]
        for _ in range(self.n_epochs):
            for d, doc in enumerate(indexed):
                dv = self.document_vectors_[d]
                for word in doc:
                    # PV-DBOW step: the DOCUMENT vector predicts this word, with
                    # `negative` sampled non-words pushed down -- identical shape
                    # to word2vec's skip-gram-negative-sampling update
                    targets = [(word, 1.0)]
                    negs = rng.choice(V, size=self.negative, p=neg_dist)
                    targets += [(int(n), 0.0) for n in negs]
                    grad = np.zeros_like(dv)
                    for t, label in targets:
                        score = 1.0 / (1.0 + np.exp(-np.clip(dv @ W_out[t], -30, 30)))
                        g = (score - label) * lr
                        grad += g * W_out[t]
                        W_out[t] -= g * dv
                    self.document_vectors_[d] -= grad
        return self

    def similarity(self, i, j):
        """Cosine similarity between two documents' learned vectors."""
        a, b = self.document_vectors_[i], self.document_vectors_[j]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def most_similar(self, i, topn=5):
        """The documents most similar to document ``i``."""
        v = self.document_vectors_
        norms = np.linalg.norm(v, axis=1) * np.linalg.norm(v[i]) + 1e-12
        sims = (v @ v[i]) / norms
        order = np.argsort(sims)[::-1]
        return [(int(j), float(sims[j])) for j in order if j != i][:topn]


__all__ = ["Doc2Vec"]
