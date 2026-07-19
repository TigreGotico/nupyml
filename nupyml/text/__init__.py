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
from ._language2 import (UnigramTokenizer, beam_search, word_movers_distance,
                         AveragedPerceptronTagger, bleu, rouge_n, rouge_l)
from ._language3 import (text_tiling, pmi_collocations, SpellCorrector, chrf,
                         meteor, HMMTagger)
from .plsa import PLSA
from .aho_corasick import AhoCorasick
from .phonetic import soundex, metaphone
from .yake import yake_keywords
from .langid import LanguageDetector
from .good_turing import good_turing_smoothing

__all__ = ["KneserNeyLM", "TextRank", "LexRank", "rake_keywords", "MEMM",
           "UnigramTokenizer", "beam_search", "word_movers_distance",
           "AveragedPerceptronTagger", "bleu", "rouge_n", "rouge_l",
           "text_tiling", "pmi_collocations", "SpellCorrector", "chrf",
           "meteor", "HMMTagger",
           "PLSA", "AhoCorasick", "soundex", "metaphone", "yake_keywords",
           "LanguageDetector", "good_turing_smoothing"]
