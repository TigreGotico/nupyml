"""J8b: the deep belief network -- greedy layer-wise RBM pretraining + readout.

The stack must pretrain one RBM per layer, expose a top representation of the
configured size, and classify digits well above chance from that representation.
"""
import numpy as np
import pytest

from sklearn.datasets import load_digits

from nupyml.nn import DeepBeliefNetwork
from nupyml.model_selection import train_test_split
from nupyml.metrics import accuracy_score


def test_deep_belief_network_pretrains_and_classifies():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    dbn = DeepBeliefNetwork(hidden_layer_sizes=(64, 32), n_iter=30,
                            random_state=0).fit(Xtr, ytr)
    # one RBM per hidden layer, each stacked on the previous representation
    assert len(dbn.rbms_) == 2
    assert dbn._transform_stack(Xte[:5]).shape[1] == 32       # top layer size
    assert accuracy_score(yte, dbn.predict(Xte)) > 0.5        # well above 1/10 chance
    proba = dbn.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)
