"""Word embeddings and tokenizers: turning text into vectors that carry meaning.

THE IDEA THAT STARTED IT
------------------------
"You shall know a word by the company it keeps" (Firth). Words that appear in
similar CONTEXTS mean similar things -- so if you learn a vector for each word
that predicts its neighbours, words with similar neighbours end up with similar
vectors. Meaning falls out of co-occurrence statistics, with no dictionary and no
labels. The famous consequence is arithmetic on meaning: ``king - man + woman``
lands near ``queen``, because the vectors encode the regularities of usage.

TWO ROUTES TO THE SAME PLACE
----------------------------
* ``Word2Vec`` (skip-gram) -- PREDICTIVE. Slide a window over the text and train
  vectors to predict each word's neighbours, one context at a time. Local, online,
  and the model that made embeddings famous.
* ``GloVe`` -- COUNT-BASED. Build the global word-co-occurrence matrix once and
  factor it so that vector dot products match log co-occurrence. Same destination,
  reached by fitting a statistic of the whole corpus rather than streaming
  windows. The two were long argued to be secretly equivalent.

TOKENIZATION: THE STEP BEFORE ANY OF IT
---------------------------------------
Before text is vectors it is TOKENS, and where you cut matters. Split on whitespace
and every rare or misspelled word is "unknown"; split on characters and sequences
become hopelessly long. ``BPE`` (byte-pair encoding) finds the middle: learn a
vocabulary of frequent SUBWORD pieces, so "unhappiness" becomes "un + happ + iness"
-- common words stay whole, rare ones decompose into known parts, and nothing is
ever truly out-of-vocabulary. This is how every modern language model tokenizes,
and it is the humble, essential step the embeddings sit on top of.

Mikolov et al. (2013); Pennington, Socher & Manning (2014); Sennrich et al. (2016).
"""
from .word2vec import Word2Vec, GloVe
from .tokenizer import BPETokenizer, WordPieceTokenizer
from .guided import (CategoricalVectorizer, LabelGuidedEmbeddings,
                     EntityEmbeddingEncoder)

__all__ = ["Word2Vec", "GloVe", "BPETokenizer", "WordPieceTokenizer",
           "CategoricalVectorizer", "LabelGuidedEmbeddings",
           "EntityEmbeddingEncoder"]
