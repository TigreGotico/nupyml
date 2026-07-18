"""Implementations for the ``text`` package."""
import re
from collections import defaultdict, Counter

import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state
from ..graph import pagerank


_WORD = re.compile(r"[A-Za-z']+")
_STOPWORDS = set("""a an the and or but if then of to in on at for with as by is are
was were be been being this that these those it its i you he she we they them his
her their our your from not no do does did has have had will would can could should
than so such into over under out up down about""".split())


def _tokenize(text):
    return _WORD.findall(text.lower())


class KneserNeyLM(BaseEstimator):
    """The n-gram smoothing that counts CONTEXTS, not occurrences (Kneser & Ney).

    Add-one smoothing wastes probability uniformly; Kneser-Ney asks a sharper
    question. The unigram fallback is not "how often is 'Francisco'?" (often -- it
    always follows 'San') but "how many DIFFERENT words does it follow?" (almost
    none). That CONTINUATION probability is what should back off a rare bigram, and
    it is why Kneser-Ney is the strongest n-gram smoothing. Absolute discounting
    subtracts a fixed ``discount`` from each seen count and redistributes the freed
    mass by continuation probability. Interpolated bigram model here.
    """

    def __init__(self, discount=0.75):
        self.discount = discount

    def fit(self, sentences):
        self.bigrams_ = defaultdict(Counter)
        self.unigrams_ = Counter()
        self.continuations_ = defaultdict(set)      # word -> set of preceding words
        self.followers_ = defaultdict(set)          # word -> set of following words
        for sent in sentences:
            toks = sent if isinstance(sent, list) else _tokenize(sent)
            for w1, w2 in zip(toks, toks[1:]):
                self.bigrams_[w1][w2] += 1
                self.followers_[w1].add(w2)
                self.continuations_[w2].add(w1)
            self.unigrams_.update(toks)
        self.vocab_ = set(self.unigrams_)
        self.n_bigram_types_ = sum(len(v) for v in self.bigrams_.values())
        return self

    def _p_continuation(self, w):
        # P_cont(w) = (# distinct words preceding w) / (# distinct bigram types)
        return len(self.continuations_.get(w, ())) / max(self.n_bigram_types_, 1)

    def prob(self, w1, w2):
        d = self.discount
        c12 = self.bigrams_.get(w1, {}).get(w2, 0)
        c1 = self.unigrams_.get(w1, 0)
        if c1 == 0:
            return self._p_continuation(w2) or 1e-10
        first = max(c12 - d, 0) / c1
        lam = d * len(self.followers_.get(w1, ())) / c1     # back-off weight
        return first + lam * self._p_continuation(w2)

    def perplexity(self, sentences):
        logp, n = 0.0, 0
        for sent in sentences:
            toks = sent if isinstance(sent, list) else _tokenize(sent)
            for w1, w2 in zip(toks, toks[1:]):
                logp += np.log(max(self.prob(w1, w2), 1e-12)); n += 1
        return float(np.exp(-logp / max(n, 1)))


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


def rake_keywords(text, top_k=5, stopwords=None):
    """Extract keyphrases from word co-occurrence, no training (Rose et al., 2010).

    RAKE (Rapid Automatic Keyword Extraction) splits text at stopwords and
    punctuation into candidate PHRASES, then scores each word by
    ``degree / frequency`` -- degree counting how many words it co-occurs with
    inside phrases, so words that appear in longer, richer phrases score higher. A
    phrase's score is the sum of its words'. Fast, unsupervised, and surprisingly
    competitive. Returns the top-``k`` phrases with scores.
    """
    stop = _STOPWORDS if stopwords is None else set(stopwords)
    # candidate phrases are maximal runs of content words, broken by a stopword or
    # any punctuation (words and punctuation are the only tokens we keep)
    phrases = []
    cur = []
    for tok in re.findall(r"[a-z']+|[.,!?;:()\"]", text.lower()):
        if _WORD.fullmatch(tok) and tok not in stop:
            cur.append(tok)
        else:                                       # stopword or punctuation -> break
            if cur:
                phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    freq = Counter(); degree = Counter()
    for ph in phrases:
        for w in ph:
            freq[w] += 1
            degree[w] += len(ph) - 1                 # co-occurrences within phrase
    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}
    scored = [(" ".join(ph), sum(word_score[w] for w in ph)) for ph in phrases]
    # dedupe keeping the max score per phrase
    best = {}
    for p, s in scored:
        best[p] = max(best.get(p, 0), s)
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


class MEMM(BaseEstimator):
    """A discriminative sequence tagger: maximum-entropy + Markov (McCallum, 2000).

    An HMM is generative -- it models ``P(word | tag)`` and cannot easily use rich,
    overlapping features of the word (suffix, capitalisation, neighbours). A MEMM
    turns it around: a maximum-entropy (softmax) classifier directly models
    ``P(tag_t | features(word_t), tag_{t-1})``, so any features are allowed, and
    the Markov dependence on the previous tag is just another feature. Decoding
    still uses Viterbi over those local conditionals. (It has the "label bias"
    quirk that the CRF later fixed -- see ``sequence.LinearChainCRF``.)
    """

    def __init__(self, lr=0.5, epochs=200, l2=1e-3, random_state=None):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state

    def _feats(self, word, prev_tag):
        wi = self.word_idx_.get(word, len(self.word_idx_))     # OOV bucket
        f = np.zeros(self.n_features_)
        f[wi] = 1.0
        f[len(self.word_idx_) + 1 + prev_tag] = 1.0            # previous tag
        f[-1] = 1.0                                            # bias
        return f

    def fit(self, sentences, tag_seqs):
        words = sorted({w for s in sentences for w in s})
        tags = sorted({t for ts in tag_seqs for t in ts})
        self.word_idx_ = {w: i for i, w in enumerate(words)}
        self.tags_ = tags
        self.tag_idx_ = {t: i for i, t in enumerate(tags)}
        n_tags = len(tags)
        # feature layout: |vocab|+1 word slots, then |tags| prev-tag slots, then bias
        self.n_features_ = len(words) + 1 + n_tags + 1
        rng = check_random_state(self.random_state)
        W = np.zeros((n_tags, self.n_features_))
        # build training examples (features, gold tag)
        Xf, yf = [], []
        for s, ts in zip(sentences, tag_seqs):
            prev = 0
            for w, t in zip(s, ts):
                Xf.append(self._feats(w, prev))
                yf.append(self.tag_idx_[t])
                prev = self.tag_idx_[t]
        Xf = np.array(Xf); yf = np.array(yf)
        onehot = np.eye(n_tags)[yf]
        for _ in range(self.epochs):                # softmax regression by GD
            logits = Xf @ W.T
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            grad = (P - onehot).T @ Xf / len(Xf) + self.l2 * W
            W -= self.lr * grad
        self.W_ = W
        return self

    def _local_logprob(self, word, prev_tag):
        f = self._feats(word, prev_tag)
        logits = self.W_ @ f
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return np.log(p + 1e-12)

    def predict(self, sentence):
        """Viterbi decode the most likely tag sequence."""
        n_tags = len(self.tags_)
        T = len(sentence)
        delta = np.full((T, n_tags), -np.inf)
        back = np.zeros((T, n_tags), dtype=int)
        delta[0] = self._local_logprob(sentence[0], 0)
        for t in range(1, T):
            for prev in range(n_tags):
                lp = self._local_logprob(sentence[t], prev)
                cand = delta[t - 1, prev] + lp
                better = cand > delta[t]
                delta[t][better] = cand[better]
                back[t][better] = prev
        tags = np.empty(T, dtype=int)
        tags[-1] = int(delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            tags[t] = back[t + 1, tags[t + 1]]
        return [self.tags_[i] for i in tags]


__all__ = ["KneserNeyLM", "TextRank", "LexRank", "rake_keywords", "MEMM"]
