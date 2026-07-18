"""L2: ensembles v4 -- random subspace, negative correlation learning, GrowNet,
snapshot ensemble.

The random subspace method beats a single tree by decorrelating members via feature
subsets; NCL fits a regression; GrowNet improves as boosting stages are added; the
snapshot ensemble beats its average member (the diversity is free from the cyclic
learning-rate schedule).
"""
import numpy as np
import pytest

from sklearn.datasets import make_classification, make_friedman1

from nupyml.ensemble import (RandomSubspaceClassifier, NegativeCorrelationLearning,
                             GrowNet, SnapshotEnsemble)
from nupyml.tree import DecisionTreeClassifier
from nupyml.model_selection import train_test_split
from nupyml.metrics import accuracy_score, r2_score


def _reg():
    X, y = make_friedman1(n_samples=300, random_state=0)
    y = (y - y.mean()) / y.std()
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_random_subspace_beats_single_tree():
    X, y = make_classification(n_samples=400, n_features=20, n_informative=8,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rs = RandomSubspaceClassifier(n_estimators=25, max_features=0.5,
                                  random_state=0).fit(Xtr, ytr)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, rs.predict(Xte)) > accuracy_score(yte,
                                                                 tree.predict(Xte))


def test_negative_correlation_learning_fits():
    Xtr, Xte, ytr, yte = _reg()
    ncl = NegativeCorrelationLearning(n_estimators=5, hidden=16, lam=0.5,
                                      epochs=400, random_state=0).fit(Xtr, ytr)
    assert r2_score(yte, ncl.predict(Xte)) > 0.4


def test_grownet_improves_with_stages():
    Xtr, Xte, ytr, yte = _reg()
    g5 = GrowNet(n_stages=5, hidden=16, random_state=0).fit(Xtr, ytr)
    g30 = GrowNet(n_stages=30, hidden=16, random_state=0).fit(Xtr, ytr)
    assert r2_score(yte, g30.predict(Xte)) > r2_score(yte, g5.predict(Xte))
    assert r2_score(yte, g30.predict(Xte)) > 0.4


def test_snapshot_ensemble_beats_average_member():
    Xtr, Xte, ytr, yte = _reg()
    se = SnapshotEnsemble(n_snapshots=6, hidden=32, cycle_epochs=100, lr_max=0.05,
                          random_state=0).fit(Xtr, ytr)
    ens = r2_score(yte, se.predict(Xte))
    members = [r2_score(yte, se._predict_with(s, Xte)) for s in se.snapshots_]
    assert ens >= np.mean(members)                       # averaging the free ensemble helps
