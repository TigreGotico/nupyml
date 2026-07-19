"""TextRank with TF-IDF cosine and a threshold graph (Erkan & Radev, 2004)."""
import re
import numpy as np
from ..base import BaseEstimator
from ..graph import pagerank
from .text_rank import TextRank


def _tokenize(text):
    return _WORD.findall(text.lower())


class _GraphSummarizer(BaseEstimator):
    def _rank(self, sim):
        np.fill_diagonal(sim, 0.0)
        row = sim.sum(axis=1, keepdims=True)
        A = np.divide(sim, row, out=np.zeros_like(sim), where=row > 0)
        return pagerank(A.T)                        # stationary sentence importance

    def summary(self, sentences, k=3):
        self.fit(sentences)
        order = np.argsort(self.scores_)[::-1][:k]
        return [sentences[i] for i in sorted(order)]


_WORD = re.compile(r"[A-Za-z']+")


class LexRank(_GraphSummarizer):
    """TextRank with TF-IDF cosine and a threshold graph (Erkan & Radev, 2004).

    LexRank sharpens TextRank's similarity: sentences are TF-IDF vectors and edges
    are their COSINE similarity, thresholded so only genuinely-related sentences
    connect. Running PageRank on that graph gives "lexical centrality" -- the
    sentences most representative of the document's central content. The TF-IDF
    weighting downplays common words that word-overlap would over-count.
    """

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, sentences):
        docs = [_tokenize(s) for s in sentences]
        vocab = sorted({w for d in docs for w in d})
        idx = {w: i for i, w in enumerate(vocab)}
        n, V = len(sentences), len(vocab)
        tf = np.zeros((n, V))
        for i, d in enumerate(docs):
            for w in d:
                tf[i, idx[w]] += 1
        df = (tf > 0).sum(axis=0)
        idf = np.log((n + 1) / (df + 1)) + 1
        X = tf * idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)
        sim = Xn @ Xn.T
        sim[sim < self.threshold] = 0.0             # keep only strong edges
        self.scores_ = self._rank(sim)
        return self


__all__ = ["LexRank"]
