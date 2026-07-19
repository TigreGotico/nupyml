"""A fast discriminative sequence tagger with weight averaging (Collins, 2002)."""
from collections import defaultdict, Counter
import numpy as np


class AveragedPerceptronTagger:
    """A fast discriminative sequence tagger with weight averaging (Collins, 2002).

    The structured perceptron tags a sequence, and on any mistake nudges the weights
    toward the gold features and away from the predicted ones -- no probabilities,
    just error-driven updates, which makes it very fast to train. Its one crucial
    trick is AVERAGING: the final weights are the average of every intermediate
    weight vector over training, which cancels the late-training thrashing a raw
    perceptron suffers and gives most of the benefit of a margin method for almost
    no cost. Features: current word, previous tag, suffixes.
    """

    def __init__(self, epochs=10, random_state=None):
        self.epochs = epochs
        self.random_state = random_state

    def _features(self, words, i, prev_tag):
        w = words[i]
        return [f"w={w}", f"prev={prev_tag}", f"suf={w[-3:]}", f"pre={w[:3]}",
                f"cap={w[0].isupper()}", "bias"]

    def fit(self, sentences, tag_seqs):
        self.tags_ = sorted({t for ts in tag_seqs for t in ts})
        weights = defaultdict(float)
        totals = defaultdict(float)
        rng = np.random.RandomState(self.random_state)
        c = 0
        order = list(range(len(sentences)))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for si in order:
                words, gold = sentences[si], tag_seqs[si]
                prev = "START"
                for i in range(len(words)):
                    feats = self._features(words, i, prev)
                    pred = self._score(feats, weights)
                    if pred != gold[i]:
                        for f in feats:                  # perceptron update
                            weights[(f, gold[i])] += 1
                            weights[(f, pred)] -= 1
                    for f in feats:
                        totals[(f, gold[i])] += weights[(f, gold[i])]
                        totals[(f, pred)] += weights[(f, pred)]
                    c += 1
                    prev = gold[i]
        self.weights_ = {k: v / c for k, v in totals.items()}   # averaged weights
        return self

    def _score(self, feats, weights):
        best, best_s = self.tags_[0], -np.inf
        for t in self.tags_:
            s = sum(weights.get((f, t), 0.0) for f in feats)
            if s > best_s:
                best_s, best = s, t
        return best

    def predict(self, words):
        prev = "START"; out = []
        for i in range(len(words)):
            feats = self._features(words, i, prev)
            t = self._score(feats, self.weights_)
            out.append(t); prev = t
        return out


__all__ = ["AveragedPerceptronTagger"]
