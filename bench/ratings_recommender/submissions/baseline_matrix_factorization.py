"""Matrix factorization by SGD -- the workhorse latent-factor recommender.

Learn a short vector per user and per item so their dot product (plus biases)
reconstructs the observed ratings; unseen cells are then just more dot products.
Recovers the low-rank structure the task is built from.
"""
import numpy as np

from nupyml.recommend import MatrixFactorization


def solve(X_train, y_train, X_test):
    triples = [(int(u), int(i), float(r))
               for (u, i), r in zip(X_train, y_train)]
    model = MatrixFactorization(n_factors=8, n_epochs=80, learning_rate=0.02,
                                reg=0.05, random_state=0).fit(triples)
    mean = float(np.mean(y_train))
    def pred(u, i):
        try:
            return model.predict(int(u), int(i))
        except (IndexError, KeyError):
            return mean            # user/item unseen in training
    return np.array([pred(u, i) for u, i in X_test])
