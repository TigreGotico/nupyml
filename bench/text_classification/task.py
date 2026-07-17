"""Text classification: label short documents by topic.

The submission receives RAW documents (arrays of strings) and must do its own
vectorisation -- so it exercises the whole text pipeline (tokenise, weight,
classify), not just a fitted feature matrix. Three topics with overlapping but
skewed vocabularies. Scored by accuracy.
"""
import numpy as np

from nupyml.metrics import accuracy_score

KIND = "supervised"
GOAL = "Classify short text documents into one of three topics; scored by accuracy."
METRIC = "accuracy"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.75


def load():
    rng = np.random.RandomState(0)
    # each topic favours some words but shares a common vocabulary (overlap)
    topics = {
        0: ["match", "score", "team", "player", "win", "goal", "season", "game"],
        1: ["market", "stock", "price", "trade", "bank", "profit", "shares", "growth"],
        2: ["film", "movie", "actor", "scene", "director", "role", "story", "award"],
    }
    common = ["the", "a", "and", "of", "in", "to", "is", "on", "for", "with"]
    docs, labels = [], []
    for label, words in topics.items():
        for _ in range(150):
            n = rng.randint(6, 12)                    # shorter docs -> less signal
            toks = []
            for _ in range(n):
                r = rng.rand()
                if r < 0.33:
                    toks.append(words[rng.randint(len(words))])   # topical word
                elif r < 0.43:
                    # a word borrowed from ANOTHER topic (cross-topic confusion)
                    other = topics[(label + 1 + rng.randint(2)) % 3]
                    toks.append(other[rng.randint(len(other))])
                else:
                    toks.append(common[rng.randint(len(common))])  # filler
            docs.append(" ".join(toks))
            labels.append(label)
    docs = np.array(docs, dtype=object)
    labels = np.array(labels)
    perm = rng.permutation(len(labels))
    docs, labels = docs[perm], labels[perm]
    n_tr = int(0.7 * len(labels))
    return docs[:n_tr], labels[:n_tr], docs[n_tr:], labels[n_tr:]


def metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
