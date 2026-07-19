"""A discriminative sequence tagger: maximum-entropy + Markov (McCallum, 2000)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


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


__all__ = ["MEMM"]
