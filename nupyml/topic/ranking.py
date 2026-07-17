"""BM25: the ranking function that ran search for two decades.

THE PROBLEM WITH NAIVE tf-idf RANKING
-------------------------------------
Score a document for a query by summing tf-idf over the query terms and you get
two things wrong:

* **Term frequency should SATURATE.** A document that mentions "jaguar" 100 times
  is not 100x more relevant than one that mentions it 10 times -- after a handful
  of occurrences the marginal evidence flattens. Raw tf grows linearly and
  over-rewards keyword stuffing.
* **Long documents cheat.** A long document contains more of every word by sheer
  length, so raw counts favour it unfairly. Relevance should be judged relative
  to the document's length.

BM25 FIXES BOTH, WITH TWO KNOBS
-------------------------------
    score(doc, query) = sum over query terms of
        idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |doc| / avg_len))

* ``k1`` controls SATURATION: the ``tf/(tf + k1...)`` form rises then flattens,
  so the 10th occurrence adds far less than the 1st. ``k1 -> infinity`` recovers
  raw linear tf; ``k1 = 0`` makes presence binary.
* ``b`` controls LENGTH NORMALISATION: ``b = 1`` fully divides by relative
  length, ``b = 0`` ignores length entirely; 0.75 is the tuned default.

Those two corrections are the entire difference between tf-idf and BM25, and they
were enough to make BM25 the default relevance function in Lucene/Elasticsearch
and the strong classical baseline that neural rankers are still measured against.

Robertson & Zaragoza (2009).
"""
import numpy as np

from ..base import BaseEstimator


class BM25(BaseEstimator):
    """Rank documents by relevance to a query.

    ``fit`` takes a list of tokenised documents (lists of terms). ``score`` and
    ``rank`` take a tokenised query.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents):
        self.docs_ = [list(d) for d in documents]
        self.doc_len_ = np.array([len(d) for d in self.docs_], dtype=float)
        self.avg_len_ = self.doc_len_.mean()
        n = len(self.docs_)

        # document frequency of each term, then its idf
        self.term_freqs_ = []           # per-doc {term: count}
        df = {}
        for d in self.docs_:
            tf = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            self.term_freqs_.append(tf)
            for w in tf:
                df[w] = df.get(w, 0) + 1
        # the BM25 idf: rarer query terms carry more weight (the +0.5 smoothing
        # is the Robertson-Sparck-Jones form, which stays sane for common terms)
        self.idf_ = {w: np.log((n - dfw + 0.5) / (dfw + 0.5) + 1.0)
                     for w, dfw in df.items()}
        return self

    def score(self, query):
        """BM25 score of every document for the query."""
        scores = np.zeros(len(self.docs_))
        for i, tf in enumerate(self.term_freqs_):
            length_norm = 1 - self.b + self.b * self.doc_len_[i] / self.avg_len_
            for w in query:
                if w not in tf:
                    continue
                freq = tf[w]
                # the saturating term-frequency component, length-normalised
                numer = freq * (self.k1 + 1)
                denom = freq + self.k1 * length_norm
                scores[i] += self.idf_.get(w, 0.0) * numer / denom
        return scores

    def rank(self, query, top_n=None):
        """Document indices sorted by relevance, most relevant first."""
        order = np.argsort(self.score(query))[::-1]
        return order[:top_n] if top_n else order


__all__ = ["BM25"]
