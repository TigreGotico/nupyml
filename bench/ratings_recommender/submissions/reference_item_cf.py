"""Item-based collaborative filtering -- predict from similar ITEMS.

A rating is the similarity-weighted average of the user's ratings on the items
most similar to the target. The transparent, training-free neighbourhood method.
"""
import numpy as np

from nupyml.recommend import ItemBasedCF


def solve(X_train, y_train, X_test):
    triples = [(int(u), int(i), float(r))
               for (u, i), r in zip(X_train, y_train)]
    model = ItemBasedCF(k=20).fit(triples)
    mean = float(np.mean(y_train))
    def pred(u, i):
        try:
            return model.predict(int(u), int(i))
        except (IndexError, KeyError):
            return mean            # user/item unseen in training
    return np.array([pred(u, i) for u, i in X_test])
