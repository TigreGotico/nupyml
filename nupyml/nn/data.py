"""Minimal batching utilities."""
import numpy as np

from ..utils import check_random_state


class DataLoader:
    """Iterate over (X, y) [or just X] in shuffled mini-batches."""

    def __init__(self, X, y=None, batch_size=32, shuffle=True, random_state=None):
        self.X = np.asarray(X)
        self.y = None if y is None else np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = check_random_state(random_state)

    def __iter__(self):
        n = len(self.X)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch = idx[start:start + self.batch_size]
            if self.y is None:
                yield self.X[batch]
            else:
                yield self.X[batch], self.y[batch]

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size


__all__ = ["DataLoader"]
