"""Text and language: smoothing, extractive summarisation, keywords, tagging.

Classic NLP algorithms that are neither topic models (see ``topic``) nor
embeddings (see ``embed``):

* ``KneserNeyLM`` -- the n-gram smoothing that actually works, built on how many
  DIFFERENT contexts a word completes, not just how often it appears.
* ``TextRank`` / ``LexRank`` -- summarise by running PageRank over a graph of
  sentences: the central sentences are the summary.
* ``rake_keywords`` -- extract keyphrases with no training, from word co-occurrence
  within phrases.
* ``MEMM`` -- a discriminative sequence tagger: a maximum-entropy model that
  conditions each tag on the word and the PREVIOUS tag, decoded with Viterbi.
"""
from ._language import (KneserNeyLM, TextRank, LexRank, rake_keywords, MEMM)

__all__ = ["KneserNeyLM", "TextRank", "LexRank", "rake_keywords", "MEMM"]
