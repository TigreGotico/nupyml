"""Sequence mining: classify sequences by the pattern planted in them.

Each sample is a symbol sequence. Class 0 hides the ordered subsequence A->B->C among
filler; class 1 hides D->E->F. The discriminative signal is a SEQUENTIAL pattern, not a
bag of symbols, so a method that mines subsequences should beat one that ignores order.
Scored by accuracy. Sequences are passed as an object array of Python lists.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = "Classify sequences by their planted sequential pattern; accuracy."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.80

_FILLER = list("GHIJKL")


def _make(pattern, rng, length=10):
    seq = list(rng.choice(_FILLER, length))
    pts = sorted(rng.choice(range(length), len(pattern), replace=False))
    for p, sym in zip(pts, pattern):
        seq[p] = sym                                       # embed the pattern in order
    return seq


def _obj_array(list_of_seqs):
    arr = np.empty(len(list_of_seqs), dtype=object)
    for i, s in enumerate(list_of_seqs):
        arr[i] = s
    return arr


def load():
    rng = np.random.RandomState(0)
    seqs, labels = [], []
    for _ in range(150):
        seqs.append(_make(["A", "B", "C"], rng)); labels.append(0)
        seqs.append(_make(["D", "E", "F"], rng)); labels.append(1)
    X = _obj_array(seqs); y = np.array(labels)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
