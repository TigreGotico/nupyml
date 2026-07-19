"""Vectorise arbitrary features into a fixed size by HASHING (Weinberger, 2009)."""
import numpy as np


def feature_hashing(records, n_features=256, seed=0):
    """Vectorise arbitrary features into a fixed size by HASHING (Weinberger, 2009).

    A bag of raw features (words, categories, id strings) can have millions of
    distinct keys, and a normal one-hot needs a growing vocabulary you must store
    and keep in sync. The hashing trick drops the vocabulary: hash each feature key
    to one of ``n_features`` columns and add its value there, with a second hash
    giving a SIGN so that random collisions tend to cancel rather than accumulate.
    Fixed memory, no fitting, streaming-friendly -- at the cost of occasional
    collisions. ``records`` is a list of dicts or of token lists.
    """
    out = np.zeros((len(records), n_features))
    for i, rec in enumerate(records):
        items = rec.items() if isinstance(rec, dict) else ((tok, 1.0) for tok in rec)
        for key, val in items:
            h = hash((seed, str(key)))
            col = h % n_features
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0    # sign hash cancels collisions
            out[i, col] += sign * val
    return out


__all__ = ["feature_hashing"]
