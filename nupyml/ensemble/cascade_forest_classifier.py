"""Deep Forest: stack forest LAYERS like a neural net (Zhou & Feng, 2017)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


class CascadeForestClassifier(BaseEstimator, ClassifierMixin):
    """Deep Forest: stack forest LAYERS like a neural net (Zhou & Feng, 2017).

    "Deep" need not mean neural. gcForest stacks LAYERS of forests: each layer is
    an ensemble of random forests, and its class-probability outputs are CONCATENATED
    to the original features and fed to the next layer -- representation learning by
    forests instead of neurons. It grows layers until validation accuracy stops
    improving, so the depth is chosen automatically (no backprop, few
    hyperparameters). Here each layer holds a few random forests.
    """

    def __init__(self, n_layers=5, n_forests=2, n_estimators=30, max_depth=None,
                 tol=1e-4, random_state=None):
        self.n_layers = n_layers
        self.n_forests = n_forests
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        from . import RandomForestClassifier
        from ..model_selection import train_test_split
        from ..metrics import accuracy_score
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3,
                                                random_state=rng)
        self.layers_ = []
        cur_tr, cur_val = Xtr, Xval
        prev_acc = -np.inf
        for _ in range(self.n_layers):
            forests, tr_out, val_out = [], [], []
            for f in range(self.n_forests):
                rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            max_features="sqrt" if f == 0 else None,
                                            random_state=rng)
                rf.fit(cur_tr, ytr)
                forests.append(rf)
                tr_out.append(rf.predict_proba(cur_tr))
                val_out.append(rf.predict_proba(cur_val))
            acc = accuracy_score(yval, self.classes_[
                np.mean(val_out, axis=0).argmax(axis=1)])
            if acc <= prev_acc + self.tol and self.layers_:
                break                                    # stop growing depth
            prev_acc = acc
            self.layers_.append(forests)
            cur_tr = np.hstack([Xtr] + tr_out)           # augment with class probs
            cur_val = np.hstack([Xval] + val_out)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        cur = X
        last = None
        for forests in self.layers_:
            outs = [f.predict_proba(cur) for f in forests]
            last = np.mean(outs, axis=0)
            cur = np.hstack([X] + outs)
        return last

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["CascadeForestClassifier"]
