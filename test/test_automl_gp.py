"""F12: sparse & multi-output GPs, ensemble selection, and BOHB.

Sparse GP must approximate a full GP with far fewer inducing points; the
multi-output GP must predict every target; ensemble selection must be competitive
with the best single library model and return valid weights; BOHB must find a
good configuration.
"""
import numpy as np
import pytest

from nupyml.gaussian_process import (SparseGaussianProcessRegressor,
                                     MultiOutputGaussianProcessRegressor, RBF)
from nupyml.ensemble import (EnsembleSelectionClassifier,
                            EnsembleSelectionRegressor, RandomForestClassifier)
from nupyml.model_selection import BOHBSearchCV, train_test_split
from nupyml.metrics import r2_score
from nupyml.datasets import make_classification, make_regression
from nupyml.linear_model import LogisticRegression, Ridge
from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.naive_bayes import GaussianNB


# --- sparse GP ------------------------------------------------------------

def test_sparse_gp_approximates_with_few_inducing_points():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, 400)).reshape(-1, 1)
    y = np.sin(2 * X.ravel()) + 0.1 * rng.randn(400)
    sgp = SparseGaussianProcessRegressor(kernel=RBF(length_scale=0.5),
                                         n_inducing=25, alpha=0.05,
                                         random_state=0).fit(X, y)
    mu, sd = sgp.predict(X, return_std=True)
    assert r2_score(y, mu) > 0.9                    # 25 inducing pts fit 400
    assert len(sgp.U_) == 25 and np.all(sd >= 0)


# --- multi-output GP ------------------------------------------------------

def test_multi_output_gp_predicts_every_target():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, 200)).reshape(-1, 1)
    Y = np.column_stack([np.sin(2 * X.ravel()), np.cos(2 * X.ravel())])
    Y += 0.1 * rng.randn(200, 2)
    mo = MultiOutputGaussianProcessRegressor(kernel=RBF(length_scale=0.5),
                                             alpha=0.05).fit(X, Y)
    P, S = mo.predict(X, return_std=True)
    assert P.shape == Y.shape and S.shape == Y.shape
    assert r2_score(Y[:, 0], P[:, 0]) > 0.9
    assert r2_score(Y[:, 1], P[:, 1]) > 0.9


# --- ensemble selection ---------------------------------------------------

def test_ensemble_selection_classifier_is_competitive():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    library = [DecisionTreeClassifier(max_depth=2, random_state=0),
               DecisionTreeClassifier(max_depth=5, random_state=0),
               GaussianNB(),
               LogisticRegression(max_iter=300)]
    es = EnsembleSelectionClassifier(library, n_rounds=40,
                                     random_state=0).fit(Xtr, ytr)
    assert np.isclose(es.weights_.sum(), 1.0) and (es.weights_ >= 0).all()
    ens_acc = np.mean(es.predict(Xte) == yte)
    best_single = max(np.mean(clone_fit(m, Xtr, ytr).predict(Xte) == yte)
                      for m in library)
    assert ens_acc >= best_single - 0.03            # competitive with the best


def clone_fit(est, X, y):
    from nupyml.base import clone
    return clone(est).fit(X, y)


def test_ensemble_selection_regressor_beats_worst_and_ties_best():
    X, y = make_regression(n_samples=500, n_features=8, n_informative=5,
                           noise=10.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    library = [Ridge(alpha=1.0), DecisionTreeRegressor(max_depth=3, random_state=0),
               DecisionTreeRegressor(max_depth=6, random_state=0)]
    es = EnsembleSelectionRegressor(library, n_rounds=40, random_state=0).fit(Xtr, ytr)
    ens = r2_score(yte, es.predict(Xte))
    singles = [r2_score(yte, clone_fit(m, Xtr, ytr).predict(Xte)) for m in library]
    assert ens >= min(singles)                      # never worse than the worst
    assert ens >= max(singles) - 0.05               # ~ as good as the best


# --- BOHB -----------------------------------------------------------------

def test_bohb_finds_a_good_configuration():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               random_state=0)
    bohb = BOHBSearchCV(
        RandomForestClassifier(random_state=0),
        {"max_depth": [2, 4, 8, None], "n_estimators": [10, 20, 40]},
        max_resource=1.0, eta=3, cv=3, random_state=0).fit(X, y)
    assert bohb.best_score_ > 0.8
    assert hasattr(bohb, "best_estimator_")
    assert set(bohb.best_params_) == {"max_depth", "n_estimators"}


def test_bohb_handles_continuous_params():
    from nupyml.svm import SVC
    X, y = make_classification(n_samples=400, n_features=8, n_informative=4,
                               random_state=0)
    bohb = BOHBSearchCV(SVC(random_state=0),
                        {"C": (0.1, 50.0), "gamma": (0.001, 1.0)},
                        max_resource=1.0, cv=3, random_state=0).fit(X, y)
    assert 0.1 <= bohb.best_params_["C"] <= 50.0
    assert bohb.best_score_ > 0.8
