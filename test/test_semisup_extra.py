"""H8: co-training, tri-training, and positive-unlabeled learning.

Co/tri-training must learn a good classifier from few labels + many unlabeled;
PU learning must recover positives from positive-and-unlabeled data and clearly
beat the naive "unlabeled = negative" classifier.
"""
import numpy as np
import pytest

from nupyml.semi_supervised import CoTraining, TriTraining, PUClassifier
from nupyml.datasets import make_classification
from nupyml.linear_model import LogisticRegression
from nupyml.model_selection import train_test_split


def _semi(seed=0, label_frac=0.05, class_sep=0.8):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                               n_classes=2, class_sep=class_sep, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    rng = np.random.RandomState(seed)
    yl = ytr.copy()
    yl[rng.rand(len(yl)) > label_frac] = -1
    return Xtr, yl, Xte, yte


@pytest.mark.parametrize("make", [
    lambda: CoTraining(n_iter=15, top_k=5, random_state=0),
    lambda: TriTraining(n_iter=10, random_state=0),
], ids=["cotraining", "tritraining"])
def test_semi_supervised_learner_is_accurate(make):
    Xtr, yl, Xte, yte = _semi()
    model = make().fit(Xtr, yl)
    labeled = yl != -1
    base = LogisticRegression(max_iter=500).fit(Xtr[labeled], yl[labeled])
    acc = (model.predict(Xte) == yte).mean()
    assert acc > 0.75
    # using the unlabeled data does not hurt vs training on labels alone
    assert acc >= (base.predict(Xte) == yte).mean() - 0.05


def test_pu_learning_beats_naive_and_recovers_positives():
    rng = np.random.RandomState(1)
    X, y = make_classification(n_samples=1200, n_features=8, n_informative=5,
                               n_classes=2, class_sep=1.2, random_state=1)
    s = np.zeros(len(y), int)
    pos = np.where(y == 1)[0]
    s[rng.choice(pos, int(0.4 * len(pos)), replace=False)] = 1   # label 40% of positives
    pu = PUClassifier(random_state=0).fit(X, s)
    naive = LogisticRegression(max_iter=500).fit(X, s)           # treats unlabeled as neg
    pu_acc = (pu.predict(X) == y).mean()
    naive_acc = (naive.predict(X) == y).mean()
    assert pu_acc > 0.9
    assert pu_acc > naive_acc + 0.1                              # the correction helps a lot
    assert 0 < pu.c_ <= 1                                        # a valid P(labeled|positive)
