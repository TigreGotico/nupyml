"""Baseline: the local z-score of deviation from the rolling mean."""
def solve(X):
    return X[:, 2]                                        # deviation / rolling std
