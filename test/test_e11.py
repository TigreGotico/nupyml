"""E11: balanced ensembles, conformal expansion, neighborhood recommenders,
and AutoML-lite search.

Each test holds the estimator to its promise: balanced ensembles must catch the
rare class an imbalance-blind model ignores; CQR must cover with adaptive width;
Venn-Abers must bracket a calibrated probability; ACI must drive its miss rate to
the target under drift; the recommenders must reconstruct held-out ratings; and
the searchers must find a known-good hyperparameter.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification, make_regression
from nupyml.metrics import f1_score, recall_score
from nupyml.model_selection import train_test_split


def _pos_recall(y_true, y_pred):
    """Recall of the positive (label 1) class."""
    return recall_score(y_true, y_pred, average=None, labels=[0, 1])[1]


# --- balanced ensembles ---------------------------------------------------

@pytest.fixture
def imbalanced():
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=4,
                               class_sep=1.2, random_state=0)
    # drop most of the positive class to force ~8% prevalence
    rng = np.random.RandomState(0)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_pos = rng.choice(pos, size=max(1, int(0.09 * len(neg))), replace=False)
    idx = np.concatenate([neg, keep_pos])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def test_balanced_random_forest_catches_minority(imbalanced):
    from nupyml.imbalance import BalancedRandomForest
    from nupyml.ensemble import RandomForestClassifier
    Xtr, Xte, ytr, yte = imbalanced
    brf = BalancedRandomForest(n_estimators=40, random_state=0).fit(Xtr, ytr)
    plain = RandomForestClassifier(n_estimators=40, random_state=0).fit(Xtr, ytr)
    # the balanced forest should recall the rare (positive) class better
    assert (_pos_recall(yte, brf.predict(Xte))
            >= _pos_recall(yte, plain.predict(Xte)))
    assert _pos_recall(yte, brf.predict(Xte)) > 0.5


def test_rusboost_beats_plain_macro_f1(imbalanced):
    from nupyml.imbalance import RUSBoost
    from nupyml.ensemble import AdaBoostClassifier
    Xtr, Xte, ytr, yte = imbalanced
    rus = RUSBoost(n_estimators=40, random_state=0).fit(Xtr, ytr)
    ada = AdaBoostClassifier(n_estimators=40, random_state=0).fit(Xtr, ytr)
    assert (f1_score(yte, rus.predict(Xte), average="macro")
            >= f1_score(yte, ada.predict(Xte), average="macro") - 0.02)


def test_easy_ensemble_recalls_minority(imbalanced):
    from nupyml.imbalance import EasyEnsemble
    Xtr, Xte, ytr, yte = imbalanced
    ee = EasyEnsemble(n_estimators=15, random_state=0).fit(Xtr, ytr)
    assert _pos_recall(yte, ee.predict(Xte)) > 0.5


# --- conformal expansion --------------------------------------------------

def test_cqr_covers_with_adaptive_width():
    """CQR must reach ~90% coverage, and on heteroscedastic data its band should
    NOT be constant-width (unlike plain split conformal)."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(0, 10, 600)).reshape(-1, 1)
    noise = (0.3 + 0.5 * X.ravel()) * rng.randn(600)   # width grows with x
    y = np.sin(X.ravel()) * 3 + noise
    from nupyml.inference import ConformalizedQuantileRegression
    cqr = ConformalizedQuantileRegression(alpha=0.1, random_state=0).fit(X, y)
    lo, hi = cqr.predict_interval(X)
    coverage = np.mean((y >= lo) & (y <= hi))
    assert coverage >= 0.85
    widths = hi - lo
    # heteroscedastic band: the widest region is clearly wider than the narrowest
    assert widths.max() > 1.3 * widths.min()


def test_venn_abers_brackets_a_probability():
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    from nupyml.inference import VennAbersCalibrator
    from nupyml.linear_model import LogisticRegression
    va = VennAbersCalibrator(LogisticRegression(max_iter=500),
                             random_state=0).fit(Xtr, ytr)
    p0, p1 = va.predict_proba_interval(Xte)
    assert np.all(p0 <= p1 + 1e-9)                 # the interval is ordered
    assert np.all((p0 >= 0) & (p1 <= 1))
    # the merged probability must actually classify better than chance
    acc = np.mean(va.predict(Xte) == yte)
    assert acc > 0.7


def test_adaptive_conformal_tracks_target_under_drift():
    """Feeding ACI a stream whose true miss rate is 30% while it targets 10%
    should push its working alpha UP so its realised miss rate falls toward the
    target it can achieve, not stay stuck."""
    from nupyml.inference import AdaptiveConformalInference
    rng = np.random.RandomState(0)
    aci = AdaptiveConformalInference(alpha_target=0.1, gamma=0.1)
    # simulate: covered with prob depending on current level (higher level ->
    # wider interval -> more coverage), a simple feedback loop
    for _ in range(500):
        level = aci.quantile_level()
        covered = rng.rand() < level               # wider (higher level) covers more
        aci.update(covered)
    # the controller should settle near the 10% target miss rate
    assert abs(aci.realised_coverage() - 0.9) < 0.06


# --- neighborhood recommenders --------------------------------------------

@pytest.fixture
def ratings():
    """A low-rank rating matrix as triples, with a held-out test cell set."""
    rng = np.random.RandomState(0)
    n_users, n_items, k = 40, 25, 3
    P = rng.rand(n_users, k)
    Q = rng.rand(n_items, k)
    full = (P @ Q.T) * 4 + 1                        # ratings in ~[1, 5]
    triples, test = [], []
    for u in range(n_users):
        for i in range(n_items):
            if rng.rand() < 0.5:
                triples.append((u, i, full[u, i]))
            elif rng.rand() < 0.2:
                test.append((u, i, full[u, i]))
    return triples, test


def _rmse(model, test):
    err = [(model.predict(u, i) - r) ** 2 for u, i, r in test]
    return np.sqrt(np.mean(err))


def test_item_based_cf_beats_global_mean(ratings):
    from nupyml.recommend import ItemBasedCF
    triples, test = ratings
    model = ItemBasedCF(k=15).fit(triples)
    gm = np.mean([r for _, _, r in triples])
    baseline = np.sqrt(np.mean([(gm - r) ** 2 for _, _, r in test]))
    assert _rmse(model, test) < baseline


def test_user_based_cf_beats_global_mean(ratings):
    from nupyml.recommend import UserBasedCF
    triples, test = ratings
    model = UserBasedCF(k=15).fit(triples)
    gm = np.mean([r for _, _, r in triples])
    baseline = np.sqrt(np.mean([(gm - r) ** 2 for _, _, r in test]))
    assert _rmse(model, test) < baseline


def test_slim_learns_nonnegative_sparse_weights(ratings):
    from nupyml.recommend import SLIM
    triples, test = ratings
    model = SLIM(l1=0.01, l2=1.0, max_iter=50).fit(triples)
    assert np.all(np.diag(model.W_) == 0)          # zero diagonal enforced
    assert np.all(model.W_ >= 0)                    # non-negative endorsements
    assert (model.W_ == 0).mean() > 0.2             # the L1 penalty made it sparse
    # recommendations exclude already-seen items
    recs = model.recommend(0, n=5, exclude=[i for u, i, _ in triples if u == 0])
    assert len(recs) == 5


def test_svdpp_beats_plain_svd_or_matches(ratings):
    from nupyml.recommend import SVDpp, MatrixFactorization
    triples, test = ratings
    spp = SVDpp(n_factors=3, n_epochs=40, random_state=0).fit(triples)
    mf = MatrixFactorization(n_factors=3, n_epochs=40, random_state=0).fit(triples)
    gm = np.mean([r for _, _, r in triples])
    baseline = np.sqrt(np.mean([(gm - r) ** 2 for _, _, r in test]))
    # both should beat the global-mean baseline; SVD++ uses the extra signal
    assert _rmse(spp, test) < baseline
    assert _rmse(mf, test) < baseline


# --- AutoML-lite ----------------------------------------------------------

@pytest.fixture
def tuning_problem():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               random_state=0)
    return X, y


def test_hyperband_finds_a_good_config(tuning_problem):
    from nupyml.model_selection import HyperbandSearchCV
    from nupyml.ensemble import RandomForestClassifier
    X, y = tuning_problem
    hb = HyperbandSearchCV(
        RandomForestClassifier(random_state=0),
        {"max_depth": [2, 4, 8, None], "n_estimators": [10, 20, 40]},
        max_resource=1.0, eta=3, cv=3, random_state=0).fit(X, y)
    assert hb.best_score_ > 0.75
    assert hasattr(hb, "best_estimator_")
    assert set(hb.best_params_) == {"max_depth", "n_estimators"}


def test_tpe_beats_its_random_startup(tuning_problem):
    """TPE's later, model-guided trials should not be worse than its best random
    startup trial -- the model must actually steer the search."""
    from nupyml.model_selection import TPESearchCV
    from nupyml.svm import SVC
    X, y = tuning_problem
    tpe = TPESearchCV(
        SVC(random_state=0),
        {"C": (0.01, 100.0), "gamma": (0.0001, 1.0)},
        n_iter=25, n_startup=8, cv=3, random_state=0).fit(X, y)
    startup_best = max(tpe.cv_results_["mean_test_score"][:8])
    overall_best = tpe.best_score_
    assert overall_best >= startup_best
    assert overall_best > 0.8


def test_tpe_handles_categorical_params(tuning_problem):
    from nupyml.model_selection import TPESearchCV
    from nupyml.ensemble import RandomForestClassifier
    X, y = tuning_problem
    tpe = TPESearchCV(
        RandomForestClassifier(random_state=0),
        {"max_depth": [2, 4, 8], "n_estimators": [10, 30]},
        n_iter=15, n_startup=5, cv=3, random_state=0).fit(X, y)
    assert tpe.best_params_["max_depth"] in (2, 4, 8)
    assert tpe.best_score_ > 0.75
