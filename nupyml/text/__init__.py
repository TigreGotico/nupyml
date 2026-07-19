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
from .kneser_ney_lm import KneserNeyLM
from .text_rank import TextRank
from .lex_rank import LexRank
from .rake_keywords import rake_keywords
from .memm import MEMM
from .unigram_tokenizer import UnigramTokenizer
from .beam_search import beam_search
from .word_movers_distance import word_movers_distance
from .averaged_perceptron_tagger import AveragedPerceptronTagger
from .bleu import bleu
from .rouge_n import rouge_n
from .rouge_l import rouge_l
from .text_tiling import text_tiling
from .pmi_collocations import pmi_collocations
from .spell_corrector import SpellCorrector
from .chrf import chrf
from .meteor import meteor
from .hmm_tagger import HMMTagger
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
