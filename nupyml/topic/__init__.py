"""Topic models and text ranking: structure in a collection of documents.

THE IDEA
--------
A corpus is a pile of documents, each a bag of words. What are the recurring
THEMES, and which documents are about what? Topic models answer this without
labels: they discover a small number of "topics" -- distributions over words --
and describe each document as a mixture of them. "This document is 70% sports,
20% politics, 10% weather", where nobody defined sports, politics or weather;
they emerged from co-occurrence.

TWO ROUTES, ONE GOAL
--------------------
* ``LatentDirichletAllocation`` -- a GENERATIVE probability model: each document
  picks a mixture of topics, each word picks a topic then a word from it. Fitting
  inverts that story to recover the topics. Gives proper probability
  distributions and the famously interpretable "topic = list of words".
* ``LatentSemanticAnalysis`` -- a LINEAR-ALGEBRA route: factor the term-document
  matrix with an SVD and read topics off the singular vectors. Older, faster, no
  probabilistic story, but the same "documents live in a low-dimensional topic
  space" payoff -- and the origin of the whole idea (LSI, 1990).

Plus the tools that surround them:

* ``BM25`` -- the ranking function behind search engines: score a document's
  relevance to a query, correctly handling term frequency saturation and
  document length. Not a topic model, but the other half of "doing something
  useful with a bag of words".
* ``Doc2Vec`` -- learn a dense vector per DOCUMENT (extending word2vec), so whole
  documents can be compared by cosine similarity.
* ``topic_coherence`` -- how INTERPRETABLE a learned topic is, since perplexity
  notoriously disagrees with what humans find coherent.

Blei, Ng & Jordan (2003); Deerwester et al. (1990); Robertson & Zaragoza (2009).
"""
from .lda import LatentDirichletAllocation, topic_coherence
from .lsa import LatentSemanticAnalysis
from .ranking import BM25
from .doc2vec import Doc2Vec
from .language import (NGramLanguageModel, textrank_keywords, textrank_summary,
                      SIFEmbedding)

__all__ = [
    "LatentDirichletAllocation", "topic_coherence", "LatentSemanticAnalysis",
    "BM25", "Doc2Vec",
    "NGramLanguageModel", "textrank_keywords", "textrank_summary", "SIFEmbedding",
]
