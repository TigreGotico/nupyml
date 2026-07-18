"""The deep belief network: greedy layer-wise pretraining of stacked RBMs."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_array, check_random_state
from .rbm import BernoulliRBM


class DeepBeliefNetwork(BaseEstimator, ClassifierMixin):
    """Stack RBMs and train them ONE LAYER AT A TIME (Hinton et al., 2006).

    Before residual connections and good initialisations, deep nets could not be
    trained by backprop -- gradients vanished. The deep belief network was the
    breakthrough that reopened "deep": train a ``BernoulliRBM`` on the data, use its
    hidden activations as the input to train the NEXT RBM, and so on. Each layer
    learns, unsupervised and greedily, a higher-level representation, and the stack
    is a strong INITIALISATION that a final supervised readout (here a softmax) then
    fine-tunes. It is the historical bridge from shallow models to modern deep
    learning, and still a clean way to pretrain from unlabelled data. ``fit`` does
    the greedy pretraining then fits the readout on the top representation.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), learning_rate=0.05,
                 n_iter=20, readout_epochs=200, readout_lr=0.1, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.readout_epochs = readout_epochs
        self.readout_lr = readout_lr
        self.random_state = random_state

    def _transform_stack(self, X):
        h = X
        for rbm in self.rbms_:
            h = rbm.transform(h)                          # propagate up the stack
        return h

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        rng = check_random_state(self.random_state)
        self.classes_ = np.unique(y)
        # greedy layer-wise unsupervised pretraining
        self.rbms_ = []
        h = X
        for size in self.hidden_layer_sizes:
            rbm = BernoulliRBM(n_components=size, learning_rate=self.learning_rate,
                               n_iter=self.n_iter, random_state=rng)
            rbm.fit(h)
            h = rbm.transform(h)
            self.rbms_.append(rbm)
        # supervised softmax readout on the top hidden representation
        K = len(self.classes_)
        H = np.column_stack([h, np.ones(len(h))])
        W = np.zeros((K, H.shape[1]))
        yi = np.searchsorted(self.classes_, y)
        onehot = np.eye(K)[yi]
        for _ in range(self.readout_epochs):
            logits = H @ W.T
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            W -= self.readout_lr * ((P - onehot).T @ H) / len(H)
        self.readout_ = W
        return self

    def predict_proba(self, X):
        h = self._transform_stack(check_array(X))
        H = np.column_stack([h, np.ones(len(h))])
        logits = H @ self.readout_.T
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["DeepBeliefNetwork"]
