"""Subword tokenizers: BPE and WordPiece.

THE PROBLEM THEY SOLVE
----------------------
How do you turn text into a fixed vocabulary of tokens? Two naive answers both
fail:

* **Words** -- the vocabulary is unbounded (new words, typos, morphology), and
  anything unseen becomes a useless "unknown" token. "unhappiness" and
  "happiness" would be entirely unrelated symbols.
* **Characters** -- the vocabulary is tiny and nothing is ever unknown, but
  sequences become enormous and each token carries almost no meaning.

SUBWORDS are the resolution: a learned vocabulary of frequent PIECES, between
characters and words. Common words stay whole; rare words split into known
fragments ("unhappiness" -> "un" + "happiness" or "un" + "happ" + "iness"). The
vocabulary is bounded, nothing is truly out-of-vocabulary, and morphological
structure is captured for free. Every modern language model tokenizes this way.
"""
from collections import Counter

import numpy as np


class BPETokenizer:
    """Byte-Pair Encoding: merge the most frequent adjacent pair, repeatedly.

    THE ALGORITHM
    -------------
    Start with every word as a sequence of characters. Then repeat: find the
    most frequent adjacent pair of symbols across the whole corpus, and MERGE it
    into a new single symbol. "l" + "o" -> "lo", then "lo" + "w" -> "low", and so
    on. Each merge adds one token to the vocabulary; stop at the target size.

    The elegance is that frequency drives everything. Common letter sequences get
    merged early and become whole-word tokens; rare sequences are never merged and
    stay as small pieces. The vocabulary self-organises around what the corpus
    actually contains -- no linguistic rules, just counts. Borrowed from a 1994
    data-COMPRESSION algorithm, which is exactly what it is: compress the text into
    a small alphabet of frequent chunks.

    ENCODING A NEW WORD
    -------------------
    Apply the learned merges in the ORDER they were learned. A word never seen in
    training still encodes -- into whatever known pieces its characters merge into
    -- which is why there is no "unknown token" problem.

    Sennrich, Haddow & Birch (2016); Gage (1994).
    """

    def __init__(self, vocab_size=100, end_of_word="</w>"):
        self.vocab_size = vocab_size
        self.end_of_word = end_of_word

    def fit(self, corpus):
        """``corpus`` is a list of strings (documents or sentences)."""
        # each word as a tuple of characters, with an end marker so the tokenizer
        # can tell "est" at a word's end from "est" inside one
        word_freqs = Counter()
        for doc in corpus:
            for word in doc.split():
                word_freqs[word] += 1
        words = {tuple(list(w) + [self.end_of_word]): f
                 for w, f in word_freqs.items()}

        # the base vocabulary is every character seen
        vocab = set()
        for w in words:
            vocab.update(w)
        self.merges_ = []

        while len(vocab) < self.vocab_size:
            # count every adjacent symbol pair across the (frequency-weighted)
            # corpus, and merge the most common
            pairs = Counter()
            for word, freq in words.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges_.append(best)
            # apply the merge everywhere it occurs, forming a new symbol
            merged = best[0] + best[1]
            vocab.add(merged)
            new_words = {}
            for word, freq in words.items():
                new_word = []
                i = 0
                while i < len(word):
                    if (i < len(word) - 1 and word[i] == best[0]
                            and word[i + 1] == best[1]):
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words[tuple(new_word)] = freq
            words = new_words

        self.vocab_ = sorted(vocab)
        self.token_to_id_ = {t: i for i, t in enumerate(self.vocab_)}
        return self

    def tokenize(self, word):
        """Encode one word by replaying the learned merges in order."""
        symbols = list(word) + [self.end_of_word]
        for a, b in self.merges_:
            # apply this merge wherever the pair appears -- order matters, since
            # later merges may depend on earlier ones having happened
            merged = a + b
            i = 0
            out = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def encode(self, text):
        """Text -> list of token ids."""
        ids = []
        for word in text.split():
            for tok in self.tokenize(word):
                ids.append(self.token_to_id_.get(tok, -1))
        return ids


class WordPieceTokenizer:
    """WordPiece: like BPE, but merge by LIKELIHOOD gain, not raw frequency.

    THE ONE DIFFERENCE FROM BPE
    ---------------------------
    BPE merges the most FREQUENT pair. WordPiece merges the pair that most
    increases the training-data likelihood -- which scores a pair by::

        freq(ab) / (freq(a) * freq(b))

    rather than ``freq(ab)`` alone. The division is the whole idea: it rewards
    pairs that occur together MORE than their individual frequencies would
    predict, and penalises a pair that is common only because both parts are
    common. So WordPiece prefers genuinely BOUND pieces ("##ing") over
    coincidentally-adjacent frequent characters, giving more linguistically
    coherent subwords. It is the tokenizer BERT uses.

    (This is a compact illustration of the scoring difference, not a full
    likelihood-optimal implementation.)

    Schuster & Nakajima (2012).
    """

    def __init__(self, vocab_size=100, continuation="##"):
        self.vocab_size = vocab_size
        self.continuation = continuation

    def fit(self, corpus):
        word_freqs = Counter()
        for doc in corpus:
            for word in doc.split():
                word_freqs[word] += 1
        # mark non-initial pieces with the continuation prefix, WordPiece-style
        words = {}
        for w, f in word_freqs.items():
            chars = [w[0]] + [self.continuation + c for c in w[1:]]
            words[tuple(chars)] = f

        vocab = set()
        for w in words:
            vocab.update(w)
        self.merges_ = []

        while len(vocab) < self.vocab_size:
            pair_freq = Counter()
            sym_freq = Counter()
            for word, freq in words.items():
                for s in word:
                    sym_freq[s] += freq
                for i in range(len(word) - 1):
                    pair_freq[(word[i], word[i + 1])] += freq
            if not pair_freq:
                break
            # score by likelihood gain: co-occurrence over the product of parts
            def score(pair):
                a, b = pair
                return pair_freq[pair] / (sym_freq[a] * sym_freq[b])
            best = max(pair_freq, key=score)

            a, b = best
            # merging drops the continuation marker on the second piece
            merged = a + b.replace(self.continuation, "")
            vocab.add(merged)
            self.merges_.append(best)
            new_words = {}
            for word, freq in words.items():
                new_word, i = [], 0
                while i < len(word):
                    if (i < len(word) - 1 and word[i] == a and word[i + 1] == b):
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words[tuple(new_word)] = freq
            words = new_words

        self.vocab_ = sorted(vocab)
        self.token_to_id_ = {t: i for i, t in enumerate(self.vocab_)}
        return self


__all__ = ["BPETokenizer", "WordPieceTokenizer"]
