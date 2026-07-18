"""Matrix factorization by SGD -- learn low-rank factors from observed cells."""
import numpy as np

from nupyml.recommend import MatrixFactorization


def solve(X_train, y_train, X_test):
    triples = [(int(r), int(c), float(v)) for (r, c), v in zip(X_train, y_train)]
    m = MatrixFactorization(n_factors=6, n_epochs=100, learning_rate=0.02,
                            reg=0.02, random_state=0).fit(triples)
    mean = float(np.mean(y_train))
    def pred(r, c):
        try:
            return m.predict(int(r), int(c))
        except (IndexError, KeyError):
            return mean
    return np.array([pred(r, c) for r, c in X_test])
