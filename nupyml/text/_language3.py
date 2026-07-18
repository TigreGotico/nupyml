"""Language v4: topic segmentation, collocations, spelling correction, generation
metrics, and generative tagging.

Five more language tools. TextTiling splits a document at topic shifts. PMI finds
word pairs that go together more than chance. The Norvig corrector fixes typos with
a noisy-channel model. METEOR and chrF score generated text with recall and
character-level matching. The HMM tagger is the generative counterpart of the MEMM.
"""
import re
from collections import Counter, defaultdict

import numpy as np


_WORD = re.compile(r"[A-Za-z']+")


def _tokenize(text):
    return _WORD.findall(text.lower())


def text_tiling(text, block_size=10, gap_step=5, smoothing=1):
    """Split a document at TOPIC shifts by lexical cohesion (Hearst, 1997).

    Within a topic the vocabulary is stable; at a topic boundary it changes. TextTiling
    reads that directly: slide two adjacent windows of words across the text and, at
    each gap, measure how similar their word distributions are (a cosine). Cohesion
    dips into a VALLEY wherever the vocabulary turns over, and the deepest valleys are
    the segment boundaries -- an unsupervised segmentation using nothing but word
    overlap. Returns the token offsets where topics change.
    """
    tokens = _tokenize(text)
    n = len(tokens)
    gaps = list(range(block_size, n - block_size, gap_step))
    scores = []
    for g in gaps:
        left = Counter(tokens[g - block_size:g])
        right = Counter(tokens[g:g + block_size])
        vocab = set(left) | set(right)
        a = np.array([left[w] for w in vocab]); b = np.array([right[w] for w in vocab])
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        scores.append(cos)
    scores = np.array(scores)
    for _ in range(smoothing):                            # smooth the cohesion curve
        scores = np.convolve(scores, [0.25, 0.5, 0.25], mode="same")
    # boundaries at valleys deeper than mean - std
    boundaries = []
    thresh = scores.mean() - scores.std()
    for i in range(1, len(scores) - 1):
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1] \
                and scores[i] < thresh:
            boundaries.append(gaps[i])
    return boundaries


def pmi_collocations(corpus, min_count=2, top_k=10):
    """Find word pairs that go together MORE than chance (Church & Hanks, 1990).

    "New York" is a unit; "the cat" is not, even though "the" is far more frequent.
    Raw bigram counts confuse the two. Pointwise mutual information corrects for how
    common each word is on its own: ``PMI = log2 P(w1,w2) / (P(w1) P(w2))`` -- how
    much MORE the pair occurs than if the words were independent. High PMI pinpoints
    genuine collocations (names, idioms, technical terms). A minimum count filters the
    rare-pair PMI spikes. Returns the top pairs with their PMI.
    """
    unigrams = Counter()
    bigrams = Counter()
    for line in corpus:
        toks = _tokenize(line)
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    total_u = sum(unigrams.values())
    total_b = sum(bigrams.values())
    scores = []
    for (w1, w2), c in bigrams.items():
        if c < min_count:
            continue
        p12 = c / total_b
        p1 = unigrams[w1] / total_u
        p2 = unigrams[w2] / total_u
        scores.append(((w1, w2), np.log2(p12 / (p1 * p2))))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


class SpellCorrector:
    """Fix typos with a noisy-channel edit model (Norvig, 2007).

    A misspelling is a correct word corrupted by a few edits. To fix it, generate
    every word within one or two EDITS (insert, delete, replace, transpose) of the
    input, keep those that are REAL words (in a frequency dictionary), and return the
    most probable one -- most frequent wins, since a common word is a more likely
    intended target than a rare one. It is the whole of a working spell-checker in a
    page of code, and a clean illustration of the noisy-channel model. Trained from a
    word-frequency corpus.
    """

    def __init__(self):
        self.freq_ = Counter()

    def fit(self, corpus):
        for line in corpus:
            self.freq_.update(_tokenize(line))
        return self

    def _edits1(self, word):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
        inserts = [a + c + b for a, b in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _known(self, words):
        return {w for w in words if w in self.freq_}

    def correct(self, word):
        word = word.lower()
        candidates = (self._known([word]) or self._known(self._edits1(word))
                      or self._known({e2 for e1 in self._edits1(word)
                                      for e2 in self._edits1(e1)}) or {word})
        return max(candidates, key=lambda w: self.freq_[w])   # most probable


def _ngrams_str(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(reference, candidate, max_n=6, beta=2.0):
    """Character n-gram F-score -- robust to morphology (Popović, 2015).

    Word-level BLEU misses near-matches: "colour" vs "color" scores zero overlap.
    chrF works on CHARACTER n-grams instead, so shared stems and inflections still
    match, and it correlates better with human judgement especially for
    morphologically rich languages and for short segments. It is the mean F-score
    over character n-grams of length 1..``max_n``, with recall weighted ``beta`` times
    precision. No tokenisation needed. 1.0 is a perfect match.
    """
    ref = re.sub(r"\s+", "", reference.lower())
    cand = re.sub(r"\s+", "", candidate.lower())
    fs = []
    for n in range(1, max_n + 1):
        rg, cg = _ngrams_str(ref, n), _ngrams_str(cand, n)
        overlap = sum((rg & cg).values())
        if sum(cg.values()) == 0 or sum(rg.values()) == 0:
            continue
        p = overlap / sum(cg.values())
        r = overlap / sum(rg.values())
        if p + r == 0:
            fs.append(0.0)
        else:
            fs.append((1 + beta ** 2) * p * r / (beta ** 2 * p + r))
    return float(np.mean(fs)) if fs else 0.0


def meteor(reference, candidate, alpha=0.9, gamma=0.5, beta=3.0):
    """Score with a RECALL bias and a fragmentation penalty (Banerjee & Lavie, 2005).

    BLEU is precision-oriented and brittle on single sentences. METEOR aligns unigrams
    between candidate and reference, computes an F-mean that WEIGHTS RECALL heavily
    (``alpha``), then applies a FRAGMENTATION penalty: the fewer contiguous CHUNKS the
    matched words form, the better, so word order is rewarded without demanding exact
    n-gram matches. It correlates with human judgement markedly better than BLEU at
    the sentence level. Exact-match version here (no stem/synonym tables).
    """
    ref, cand = _tokenize(reference), _tokenize(candidate)
    if not cand or not ref:
        return 0.0
    ref_count = Counter(ref)
    matched = 0
    match_flags = []
    for w in cand:
        if ref_count[w] > 0:
            ref_count[w] -= 1; matched += 1; match_flags.append(True)
        else:
            match_flags.append(False)
    if matched == 0:
        return 0.0
    P = matched / len(cand); R = matched / len(ref)
    fmean = P * R / (alpha * P + (1 - alpha) * R)
    # count chunks: maximal runs of matched words
    chunks = 0; prev = False
    for f in match_flags:
        if f and not prev:
            chunks += 1
        prev = f
    penalty = gamma * (chunks / matched) ** beta
    return float(fmean * (1 - penalty))


class HMMTagger:
    """The generative counterpart of the MEMM: an HMM part-of-speech tagger.

    An HMM tags by modelling how the data was GENERATED: a hidden tag sequence
    (a Markov chain via a TRANSITION matrix) emits words (an EMISSION matrix,
    ``P(word | tag)``). Counting those probabilities from tagged text and decoding a
    new sentence with VITERBI gives the most likely tag sequence. Unlike the
    discriminative MEMM it cannot use arbitrary features, but it is simple, has no
    label-bias problem, and its emission model handles unseen tags of known words
    gracefully. Add-one smoothed; Viterbi decoding.
    """

    def __init__(self):
        pass

    def fit(self, sentences, tag_seqs):
        self.tags_ = sorted({t for ts in tag_seqs for t in ts})
        ti = {t: i for i, t in enumerate(self.tags_)}
        K = len(self.tags_)
        trans = np.ones((K, K)); emit = defaultdict(lambda: np.ones(K))
        init = np.ones(K)
        vocab = set()
        for words, tags in zip(sentences, tag_seqs):
            init[ti[tags[0]]] += 1
            for i, (w, t) in enumerate(zip(words, tags)):
                emit[w][ti[t]] += 1; vocab.add(w)
                if i > 0:
                    trans[ti[tags[i - 1]], ti[t]] += 1
        self.log_trans_ = np.log(trans / trans.sum(axis=1, keepdims=True))
        self.log_init_ = np.log(init / init.sum())
        tagcount = np.array([sum(emit[w][k] for w in vocab) for k in range(K)])
        self.emit_ = {w: np.log(emit[w] / tagcount) for w in vocab}
        self.log_unk_ = np.log(1.0 / tagcount)            # unseen-word emission
        self.ti_ = ti
        return self

    def predict(self, words):
        K = len(self.tags_)
        T = len(words)
        delta = np.full((T, K), -np.inf)
        back = np.zeros((T, K), int)
        delta[0] = self.log_init_ + self.emit_.get(words[0], self.log_unk_)
        for t in range(1, T):
            em = self.emit_.get(words[t], self.log_unk_)
            for j in range(K):
                scores = delta[t - 1] + self.log_trans_[:, j]
                back[t, j] = scores.argmax()
                delta[t, j] = scores.max() + em[j]
        tags = np.empty(T, int); tags[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            tags[t] = back[t + 1, tags[t + 1]]
        return [self.tags_[i] for i in tags]


__all__ = ["text_tiling", "pmi_collocations", "SpellCorrector", "chrf", "meteor",
           "HMMTagger"]
