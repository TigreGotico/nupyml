import numpy as np
import pytest

import sklearn.isotonic as skiso
import sklearn.multiclass as skmc

from nupyml.multiclass import (OneVsRestClassifier, OneVsOneClassifier,
                               OutputCodeClassifier, MultiOutputClassifier,
                               MultiOutputRegressor, ClassifierChain)
from nupyml.isotonic import IsotonicRegression, isotonic_regression
from nupyml.calibration import CalibratedClassifierCV
from nupyml.semi_supervised import (LabelPropagation, LabelSpreading,
                                    SelfTrainingClassifier)
from nupyml.ensemble import (StackingClassifier, StackingRegressor,
                             VotingRegressor, RandomForestClassifier)
from nupyml.linear_model import LogisticRegression, Ridge, LinearRegression
from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.naive_bayes import GaussianNB
from nupyml.svm import LinearSVC
from nupyml.metrics import brier_score_loss, log_loss, accuracy_score
from nupyml.datasets import make_classification, make_regression, make_blobs, make_moons
from nupyml.model_selection import train_test_split

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# multiclass meta-estimators
# ---------------------------------------------------------------------------

def test_ovr_wraps_binary_learner():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)
    clf = OneVsRestClassifier(LogisticRegression()).fit(X, y)
    assert len(clf.estimators_) == 4
    assert clf.score(X, y) > 0.9
    ref = skmc.OneVsRestClassifier(
        __import__("sklearn.linear_model", fromlist=["x"]).LogisticRegression()
    ).fit(X, y)
    assert clf.score(X, y) >= ref.score(X, y) - 0.05
    assert np.allclose(clf.predict_proba(X).sum(axis=1), 1)


def test_ovo_wraps_binary_learner():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=1)
    clf = OneVsOneClassifier(LogisticRegression()).fit(X, y)
    assert len(clf.estimators_) == 6  # 4 choose 2
    assert clf.score(X, y) > 0.9


def test_ovr_with_linear_svc_no_native_multiclass():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.8, random_state=2)
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
    assert clf.score(X, y) > 0.9


def test_output_code_classifier():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=3)
    clf = OutputCodeClassifier(LogisticRegression(), code_size=2.0,
                               random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.85
    # every pair of class codes must differ, or the two are indistinguishable
    assert clf.code_separation_ > 0
    assert len(np.unique(clf.code_book_, axis=0)) == 4
    # a signed code book is required to decode signed decision_function scores
    assert set(np.unique(clf.code_book_)) == {-1.0, 1.0}


def test_multioutput_classifier():
    X = rng.normal(size=(200, 6))
    y1 = (X[:, 0] + X[:, 1] > 0).astype(int)
    y2 = (X[:, 2] - X[:, 3] > 0).astype(int)
    Y = np.column_stack([y1, y2])
    clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=5)).fit(X, Y)
    pred = clf.predict(X)
    assert pred.shape == (200, 2)
    assert clf.score(X, Y) > 0.85
    probas = clf.predict_proba(X)
    assert len(probas) == 2 and probas[0].shape == (200, 2)


def test_multioutput_regressor():
    X = rng.normal(size=(200, 5))
    y1 = X @ np.array([3.0, -2, 1, 0, 0]) + rng.normal(scale=0.2, size=200)
    y2 = X @ np.array([0.0, 1, 0, -4, 2]) + rng.normal(scale=0.2, size=200)
    Y = np.column_stack([y1, y2])
    reg = MultiOutputRegressor(Ridge()).fit(X, Y)
    assert reg.predict(X).shape == (200, 2)
    assert reg.score(X, Y) > 0.9


def test_classifier_chain_uses_earlier_outputs():
    X = rng.normal(size=(300, 4))
    y1 = (X[:, 0] > 0).astype(int)
    y2 = ((X[:, 1] > 0) ^ (y1 == 1)).astype(int)   # depends on y1
    Y = np.column_stack([y1, y2])
    chain = ClassifierChain(LogisticRegression()).fit(X, Y)
    assert chain.predict(X).shape == (300, 2)
    # the chain sees y1 and so beats independent per-output fits on y2
    indep = MultiOutputClassifier(LogisticRegression()).fit(X, Y)
    assert chain.score(X, Y) > indep.score(X, Y)


# ---------------------------------------------------------------------------
# isotonic
# ---------------------------------------------------------------------------

def test_isotonic_regression_function_matches_sklearn():
    y = rng.normal(size=50).cumsum() + rng.normal(scale=2, size=50)
    ours = isotonic_regression(y)
    ref = skiso.isotonic_regression(y)
    assert np.allclose(ours, ref, atol=1e-8)
    assert np.all(np.diff(ours) >= -1e-12)  # monotone non-decreasing


def test_isotonic_regression_weighted_and_decreasing():
    y = rng.normal(size=40)
    w = rng.uniform(0.5, 2.0, size=40)
    ours = isotonic_regression(y, sample_weight=w)
    ref = skiso.isotonic_regression(y, sample_weight=w)
    assert np.allclose(ours, ref, atol=1e-8)
    dec = isotonic_regression(y, increasing=False)
    assert np.all(np.diff(dec) <= 1e-12)


def test_isotonic_estimator_matches_sklearn():
    X = np.sort(rng.uniform(0, 10, size=80))
    y = np.log1p(X) + rng.normal(scale=0.3, size=80)
    ours = IsotonicRegression().fit(X, y)
    ref = skiso.IsotonicRegression().fit(X, y)
    Xq = np.linspace(X.min(), X.max(), 30)
    assert np.allclose(ours.predict(Xq), ref.predict(Xq), atol=1e-6)
    # our default clips out-of-range inputs instead of returning nan
    assert np.isfinite(ours.predict([-5.0, 50.0])).all()


def test_isotonic_clipping_and_auto_direction():
    X = np.sort(rng.uniform(0, 5, size=50))
    y = -X + rng.normal(scale=0.2, size=50)
    iso = IsotonicRegression(increasing="auto").fit(X, y)
    assert iso.increasing_ is np.False_ or iso.increasing_ is False
    bounded = IsotonicRegression(y_min=0.0, y_max=1.0).fit(X, np.abs(y))
    p = bounded.predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def _overconfident_data():
    """Near-duplicate correlated features: naive Bayes double-counts the same
    evidence and comes out badly overconfident."""
    r = np.random.RandomState(8)
    n = 800
    y = r.randint(0, 2, n)
    signal = r.normal(y * 1.0, 1.5, size=n)
    X = np.column_stack([signal + r.normal(scale=0.05, size=n) for _ in range(6)])
    return train_test_split(X, y, test_size=0.5, random_state=0)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_brier_score(method):
    Xtr, Xte, ytr, yte = _overconfident_data()
    base = GaussianNB().fit(Xtr, ytr)      # famously poorly calibrated
    raw = brier_score_loss(yte, base.predict_proba(Xte)[:, 1])
    cal = CalibratedClassifierCV(GaussianNB(), method=method, cv=5).fit(Xtr, ytr)
    calibrated = brier_score_loss(yte, cal.predict_proba(Xte)[:, 1])
    assert calibrated < raw
    assert np.allclose(cal.predict_proba(Xte).sum(axis=1), 1)


def test_calibration_prefit():
    Xtr, Xte, ytr, yte = _overconfident_data()
    base = GaussianNB().fit(Xtr, ytr)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(Xte, yte)
    assert cal.predict(Xte).shape == yte.shape
    assert accuracy_score(yte, cal.predict(Xte)) > 0.55


def test_calibration_multiclass():
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=3.0, random_state=9)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    cal = CalibratedClassifierCV(GaussianNB(), cv=3).fit(Xtr, ytr)
    proba = cal.predict_proba(Xte)
    assert proba.shape == (len(Xte), 3)
    assert np.allclose(proba.sum(axis=1), 1)


def test_calibration_works_on_decision_function_only_estimator():
    Xtr, Xte, ytr, yte = _overconfident_data()
    cal = CalibratedClassifierCV(LinearSVC(random_state=0), cv=3).fit(Xtr, ytr)
    proba = cal.predict_proba(Xte)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert accuracy_score(yte, cal.predict(Xte)) > 0.55


# ---------------------------------------------------------------------------
# semi-supervised
# ---------------------------------------------------------------------------

def _partially_labeled(n=300, frac=0.1, seed=10):
    X, y = make_moons(n, noise=0.08, random_state=seed)
    y_semi = y.copy()
    r = np.random.RandomState(seed)
    unlabeled = r.uniform(size=n) > frac
    y_semi[unlabeled] = -1
    return X, y, y_semi


def test_label_propagation_recovers_labels():
    X, y, y_semi = _partially_labeled()
    lp = LabelPropagation(kernel="knn", n_neighbors=7).fit(X, y_semi)
    assert accuracy_score(y, lp.transduction_) > 0.9
    assert accuracy_score(y, lp.predict(X)) > 0.9


def test_label_spreading_recovers_labels():
    X, y, y_semi = _partially_labeled(seed=11)
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(X, y_semi)
    assert accuracy_score(y, ls.transduction_) > 0.9
    # points the knn graph cannot reach from any labeled node stay all-zero
    reachable = ls.label_distributions_[ls.reachable_]
    assert np.allclose(reachable.sum(axis=1), 1)
    assert ls.reachable_.mean() >= 0.95


def test_label_propagation_rbf_kernel():
    X, y, y_semi = _partially_labeled(n=200, frac=0.2, seed=12)
    lp = LabelPropagation(kernel="rbf", gamma=20.0).fit(X, y_semi)
    assert accuracy_score(y, lp.transduction_) > 0.85


def test_semi_supervised_beats_labeled_only_baseline():
    X, y, y_semi = _partially_labeled(n=400, frac=0.05, seed=13)
    labeled = y_semi != -1
    baseline = LogisticRegression().fit(X[labeled], y_semi[labeled])
    lp = LabelPropagation(kernel="knn", n_neighbors=7).fit(X, y_semi)
    assert accuracy_score(y, lp.predict(X)) > baseline.score(X, y)


def test_self_training_classifier():
    X, y, y_semi = _partially_labeled(n=400, frac=0.1, seed=14)
    st = SelfTrainingClassifier(LogisticRegression(), threshold=0.8).fit(X, y_semi)
    assert (st.transduction_ != -1).sum() > (y_semi != -1).sum()
    assert st.predict(X).shape == y.shape
    assert (st.labeled_iter_ >= 0).any()


def test_self_training_k_best():
    X, y, y_semi = _partially_labeled(n=200, frac=0.15, seed=15)
    st = SelfTrainingClassifier(LogisticRegression(), criterion="k_best",
                                k_best=20, max_iter=5).fit(X, y_semi)
    assert st.n_iter_ <= 5


# ---------------------------------------------------------------------------
# stacking / voting
# ---------------------------------------------------------------------------

def test_stacking_classifier_beats_weakest_member():
    X, y = make_moons(500, noise=0.3, random_state=16)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    members = [("lr", LogisticRegression()),
               ("dt", DecisionTreeClassifier(max_depth=3)),
               ("rf", RandomForestClassifier(n_estimators=20, random_state=0))]
    stack = StackingClassifier(members, cv=5).fit(Xtr, ytr)
    worst = min(LogisticRegression().fit(Xtr, ytr).score(Xte, yte),
                DecisionTreeClassifier(max_depth=3).fit(Xtr, ytr).score(Xte, yte))
    assert stack.score(Xte, yte) > worst
    assert np.allclose(stack.predict_proba(Xte).sum(axis=1), 1)


def test_stacking_passthrough_and_custom_final():
    X, y = make_classification(n_samples=300, n_features=6, random_state=17)
    stack = StackingClassifier(
        [("nb", GaussianNB()), ("dt", DecisionTreeClassifier(max_depth=3))],
        final_estimator=LogisticRegression(), passthrough=True, cv=3).fit(X, y)
    # meta features = 2 base columns + 6 passthrough features
    assert stack.transform(X).shape == (300, 8)
    assert stack.score(X, y) > 0.85


def test_stacking_regressor():
    X, y = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=18)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    stack = StackingRegressor([("ridge", Ridge()),
                               ("tree", DecisionTreeRegressor(max_depth=4))],
                              cv=5).fit(Xtr, ytr)
    assert stack.score(Xte, yte) > 0.7
    assert stack.transform(Xte).shape == (len(Xte), 2)


def test_voting_regressor_with_weights():
    X, y = make_regression(n_samples=200, n_features=4, noise=1.0, random_state=19)
    vr = VotingRegressor([("lin", LinearRegression()),
                          ("tree", DecisionTreeRegressor(max_depth=3))],
                         weights=[0.8, 0.2]).fit(X, y)
    assert vr.score(X, y) > 0.85
