"""SVD++ -- matrix factorization that also uses WHICH items a user rated.

Beyond the explicit ratings, the mere set of items a user interacted with is
informative; SVD++ folds that implicit signal into the user's effective factor.
"""
import numpy as np

from nupyml.recommend import SVDpp


def solve(X_train, y_train, X_test):
    triples = [(int(u), int(i), float(r))
               for (u, i), r in zip(X_train, y_train)]
    model = SVDpp(n_factors=8, n_epochs=80, learning_rate=0.02, reg=0.05,
                  random_state=0).fit(triples)
    mean = float(np.mean(y_train))
    def pred(u, i):
        try:
            return model.predict(int(u), int(i))
        except (IndexError, KeyError):
            return mean            # user/item unseen in training
    return np.array([pred(u, i) for u, i in X_test])
