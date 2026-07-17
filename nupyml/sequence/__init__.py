"""Sequence models and sequence distances: structure in ordered symbols.

Two problems that both refuse to treat a sequence as a bag of independent items:

* **Labelling.** Tag every element of a sequence, where the tags depend on each
  other -- ``crf.py`` and the structured perceptron. A word's part of speech
  depends on its neighbours' tags, so the labels must be decided jointly, not one
  at a time.
* **Comparing.** Measure how far apart two sequences are when they may be
  stretched, shifted, or edited relative to each other -- ``distance.py``: dynamic
  time warping and the edit distances. A plain element-wise comparison fails the
  moment the sequences are misaligned, which is almost always.

Both rest on DYNAMIC PROGRAMMING: an exponential space of alignments or label
sequences, collapsed to a polynomial table because the optimal solution
decomposes into optimal sub-solutions. It is the same engine as the HMM's
forward-backward and Viterbi, and recognising it recur is most of the point.
"""
from .crf import LinearChainCRF, StructuredPerceptron
from .distance import (dtw_distance, dtw_path, levenshtein, damerau_levenshtein,
                       hamming, jaro, jaro_winkler, longest_common_subsequence,
                       needleman_wunsch)

__all__ = [
    "LinearChainCRF", "StructuredPerceptron",
    "dtw_distance", "dtw_path", "levenshtein", "damerau_levenshtein",
    "hamming", "jaro", "jaro_winkler", "longest_common_subsequence",
    "needleman_wunsch",
]
