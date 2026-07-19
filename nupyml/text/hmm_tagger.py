"""The generative counterpart of the MEMM: an HMM part-of-speech tagger."""
from collections import Counter, defaultdict
import numpy as np


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


__all__ = ["HMMTagger"]
