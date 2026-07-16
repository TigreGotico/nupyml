import numpy as np
import pytest

from nupyml.utils.estimator_checks import check_estimator
from nupyml.base import BaseEstimator, clone

from nupyml.linear_model import (
    LogisticRegression, Ridge, LinearRegression, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, Perceptron, BayesianRidge, ARDRegression,
    HuberRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor,
    QuantileRegressor, TheilSenRegressor, RANSACRegressor,
    OrthogonalMatchingPursuit, RidgeCV, LassoCV,
)
from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.ensemble import (
    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
    BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier,
)
from nupyml.cluster import (KMeans, MiniBatchKMeans, DBSCAN, Birch,
                            AgglomerativeClustering, MeanShift)
from nupyml.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                  MaxAbsScaler, Normalizer, PolynomialFeatures,
                                  Binarizer, KBinsDiscretizer)
from nupyml.decomposition import PCA, TruncatedSVD, FastICA, KernelPCA, NMF
from nupyml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from nupyml.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                              KernelDensity)
from nupyml.svm import SVC, SVR, LinearSVC, NuSVC, NuSVR
from nupyml.discriminant import (LinearDiscriminantAnalysis,
                                 QuadraticDiscriminantAnalysis)
from nupyml.mixture import GaussianMixture, BayesianGaussianMixture
from nupyml.outlier import (IsolationForest, LocalOutlierFactor, OneClassSVM,
                            EllipticEnvelope)
from nupyml.gaussian_process import (GaussianProcessRegressor,
                                     GaussianProcessClassifier)
from nupyml.nn import MLPClassifier, MLPRegressor
from nupyml.feature_selection import VarianceThreshold, SelectKBest, f_classif
from nupyml.impute import SimpleImputer, KNNImputer
from nupyml.kernel_approximation import RBFSampler, Nystroem
from nupyml.random_projection import GaussianRandomProjection
from nupyml.datasets import load_iris

ALL_ESTIMATORS = [
    LogisticRegression(), Ridge(), LinearRegression(), Lasso(), ElasticNet(),
    SGDClassifier(), SGDRegressor(), Perceptron(), BayesianRidge(),
    ARDRegression(), HuberRegressor(), PoissonRegressor(), GammaRegressor(),
    TweedieRegressor(), QuantileRegressor(), TheilSenRegressor(),
    RANSACRegressor(), OrthogonalMatchingPursuit(), RidgeCV(), LassoCV(),
    DecisionTreeClassifier(), DecisionTreeRegressor(),
    RandomForestClassifier(n_estimators=5), RandomForestRegressor(n_estimators=5),
    ExtraTreesClassifier(n_estimators=5), BaggingClassifier(n_estimators=5),
    GradientBoostingClassifier(n_estimators=5),
    GradientBoostingRegressor(n_estimators=5),
    HistGradientBoostingClassifier(max_iter=5),
    HistGradientBoostingRegressor(max_iter=5), AdaBoostClassifier(n_estimators=5),
    KMeans(n_clusters=3), MiniBatchKMeans(n_clusters=3), DBSCAN(), Birch(),
    AgglomerativeClustering(), MeanShift(),
    StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(),
    Normalizer(), PolynomialFeatures(), Binarizer(), KBinsDiscretizer(),
    PCA(n_components=2), TruncatedSVD(n_components=2), FastICA(n_components=2),
    KernelPCA(n_components=2), NMF(n_components=2),
    GaussianNB(), MultinomialNB(), BernoulliNB(),
    KNeighborsClassifier(), KNeighborsRegressor(), KernelDensity(),
    SVC(), SVR(), LinearSVC(), NuSVC(), NuSVR(),
    LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
    GaussianMixture(n_components=2), BayesianGaussianMixture(n_components=2),
    IsolationForest(n_estimators=10), LocalOutlierFactor(), OneClassSVM(),
    EllipticEnvelope(), GaussianProcessRegressor(), GaussianProcessClassifier(),
    MLPClassifier(max_iter=5), MLPRegressor(max_iter=5),
    VarianceThreshold(), SelectKBest(f_classif, k=2), SimpleImputer(),
    KNNImputer(), RBFSampler(), Nystroem(n_components=10),
    GaussianRandomProjection(n_components=2),
]


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS,
                         ids=lambda e: type(e).__name__)
def test_estimator_conformance(estimator):
    """Every estimator honours the shared API contract."""
    passed = check_estimator(estimator)
    assert len(passed) >= 10


def test_check_estimator_catches_a_broken_estimator():
    class BadEstimator(BaseEstimator):
        def __init__(self, alpha=1.0):
            self.alpha = float(alpha) * 2      # mutating a param in __init__
        def fit(self, X, y):
            return self

    with pytest.raises(AssertionError, match="__init__ changed parameter"):
        check_estimator(BadEstimator())


def test_check_estimator_catches_fit_not_returning_self():
    class NoSelf(BaseEstimator):
        def __init__(self, a=1):
            self.a = a
        def fit(self, X, y):
            self.coef_ = 1
            return None

    with pytest.raises(AssertionError, match="must return self"):
        check_estimator(NoSelf())


def test_check_estimator_generate_only():
    pairs = check_estimator(Ridge(), generate_only=True)
    assert len(pairs) > 5
    assert all(callable(check) for _, check in pairs)


# ---------------------------------------------------------------------------
# feature names
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas", reason="pandas is an optional integration")


@pytest.fixture
def iris_df():
    d = load_iris()
    cols = [c.split(" (")[0].replace(" ", "_") for c in d.feature_names]
    return pd.DataFrame(d.data, columns=cols), d.target


def test_feature_names_in_recorded_from_dataframe(iris_df):
    df, y = iris_df
    est = StandardScaler().fit(df)
    assert list(est.feature_names_in_) == list(df.columns)


def test_no_feature_names_for_plain_numpy(iris_df):
    df, y = iris_df
    est = StandardScaler().fit(df.to_numpy())
    assert not hasattr(est, "feature_names_in_")


def test_feature_name_mismatch_raises(iris_df):
    df, y = iris_df
    est = StandardScaler().fit(df)
    renamed = df.rename(columns={df.columns[0]: "something_else"})
    with pytest.raises(ValueError, match="feature names should match"):
        est.transform(renamed)


def test_get_feature_names_out_passthrough(iris_df):
    df, y = iris_df
    est = StandardScaler().fit(df)
    assert list(est.get_feature_names_out()) == list(df.columns)


def test_get_feature_names_out_for_dimensionality_change(iris_df):
    df, y = iris_df
    pca = PCA(n_components=2).fit(df)
    names = pca.get_feature_names_out()
    assert len(names) == 2
    assert list(names) == ["pca0", "pca1"]


def test_get_feature_names_out_for_selector(iris_df):
    df, y = iris_df
    kb = SelectKBest(f_classif, k=2).fit(df, y)
    names = list(kb.get_feature_names_out())
    # a selector drops columns; it must not rename the survivors
    assert len(names) == 2
    assert set(names) <= set(df.columns)


def test_polynomial_features_reports_expanded_width(iris_df):
    df, y = iris_df
    poly = PolynomialFeatures(degree=2).fit(df)
    assert len(poly.get_feature_names_out()) == poly.transform(df).shape[1]


# ---------------------------------------------------------------------------
# set_output
# ---------------------------------------------------------------------------

def test_set_output_pandas(iris_df):
    df, y = iris_df
    out = StandardScaler().set_output(transform="pandas").fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(df.columns)
    assert out.index.equals(df.index)


def test_set_output_default_is_numpy(iris_df):
    df, y = iris_df
    out = StandardScaler().fit_transform(df)
    assert isinstance(out, np.ndarray)


def test_set_output_returns_self_and_is_chainable(iris_df):
    df, y = iris_df
    est = StandardScaler()
    assert est.set_output(transform="pandas") is est


def test_set_output_rejects_unknown_mode():
    with pytest.raises(ValueError, match="'default' or 'pandas'"):
        StandardScaler().set_output(transform="polars")


def test_set_output_uses_generated_names(iris_df):
    df, y = iris_df
    out = PCA(n_components=2).set_output(transform="pandas").fit(df).transform(df)
    assert list(out.columns) == ["pca0", "pca1"]


def test_set_output_on_selector_keeps_original_names(iris_df):
    df, y = iris_df
    kb = SelectKBest(f_classif, k=2).set_output(transform="pandas").fit(df, y)
    out = kb.transform(df)
    assert set(out.columns) <= set(df.columns)
    assert out.shape[1] == 2


def test_pipeline_with_dataframe(iris_df):
    from nupyml.pipeline import make_pipeline
    df, y = iris_df
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2),
                         LogisticRegression()).fit(df, y)
    assert pipe.score(df, y) > 0.8


# ---------------------------------------------------------------------------
# HTML repr
# ---------------------------------------------------------------------------

def test_repr_html_contains_class_and_params():
    html = Ridge(alpha=2.5)._repr_html_()
    assert "Ridge" in html
    assert "alpha" in html and "2.5" in html
    assert html.startswith("<div") and html.endswith("</div>")


def test_repr_html_shows_fitted_state():
    X, y = load_iris(return_X_y=True)
    assert "not fitted" in Ridge()._repr_html_()
    assert "fitted" in Ridge().fit(X, y)._repr_html_()
    assert "not fitted" not in Ridge().fit(X, y)._repr_html_()


def test_repr_html_escapes_parameter_values():
    est = KBinsDiscretizer(strategy="<script>")
    html = est._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
