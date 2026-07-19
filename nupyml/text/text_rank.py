"""Summarise by PageRank over a sentence-similarity graph (Mihalcea, 2004)."""
import re
import numpy as np
from ..base import BaseEstimator
from ..graph import pagerank


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


class TextRank(_GraphSummarizer):
    """Summarise by PageRank over a sentence-similarity graph (Mihalcea, 2004).

    A sentence is important if it is similar to many other important sentences --
    exactly PageRank's recursive definition of importance, applied to text. Build a
    graph whose nodes are sentences and whose edge weights are word-overlap
    similarities, run PageRank, and the highest-scoring sentences form an
    extractive summary. Unsupervised, language-agnostic, no training. Similarity
    here is normalised common-word count.
    """

    def fit(self, sentences):
        toks = [set(_tokenize(s)) for s in sentences]
        n = len(sentences)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if not toks[i] or not toks[j]:
                    continue
                overlap = len(toks[i] & toks[j])
                denom = np.log(len(toks[i]) + 1) + np.log(len(toks[j]) + 1)
                s = overlap / denom if denom > 0 else 0.0
                sim[i, j] = sim[j, i] = s
        self.scores_ = self._rank(sim)
        return self


__all__ = ["TextRank"]
