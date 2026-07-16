import numpy as np
import pytest

from nupyml.pipeline import Pipeline, make_pipeline, FeatureUnion, ColumnTransformer
from nupyml.preprocessing import StandardScaler, MinMaxScaler
from nupyml.decomposition import PCA
from nupyml.linear_model import LogisticRegression
from nupyml.naive_bayes import MultinomialNB
from nupyml.feature_extraction import CountVectorizer, TfidfVectorizer
from nupyml.model_selection import GridSearchCV, cross_val_score
from nupyml.datasets import make_classification, make_blobs
from nupyml.base import clone

DOCS = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "dogs and cats are pets",
    "stock markets rallied today",
    "investors bought stocks and bonds",
    "the market closed higher today",
]
LABELS = np.array(["pets", "pets", "pets", "finance", "finance", "finance"])


def test_count_vectorizer_basics():
    cv = CountVectorizer()
    X = cv.fit_transform(DOCS)
    assert X.shape[0] == 6
    vocab = cv.vocabulary_
    assert "cat" in vocab and "the" in vocab
    row = X[0].toarray().ravel()
    assert row[vocab["the"]] == 2
    X2 = cv.transform(["the cat"])
    assert X2[0, vocab["cat"]] == 1


def test_count_vectorizer_ngrams_and_limits():
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = cv.fit_transform(DOCS)
    assert any(" " in t for t in cv.vocabulary_)
    cv2 = CountVectorizer(max_features=5)
    cv2.fit(DOCS)
    assert len(cv2.vocabulary_) == 5


def test_tfidf_matches_sklearn():
    import sklearn.feature_extraction.text as sktext
    ours = TfidfVectorizer().fit_transform(DOCS).toarray()
    ref = sktext.TfidfVectorizer().fit_transform(DOCS).toarray()
    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=1e-10)


def test_text_classification_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])
    pipe.fit(DOCS, LABELS)
    pred = pipe.predict(["my cat likes the dog", "stocks rallied"])
    assert pred[0] == "pets" and pred[1] == "finance"


def test_pipeline_with_gridsearch():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=4,
                               random_state=0)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=5)),
        ("clf", LogisticRegression()),
    ])
    gs = GridSearchCV(pipe, {"pca__n_components": [2, 5],
                             "clf__C": [0.1, 1.0]}, cv=3)
    gs.fit(X, y)
    assert gs.best_score_ > 0.7
    assert set(gs.best_params_) == {"pca__n_components", "clf__C"}
    assert gs.predict(X).shape == (200,)


def test_pipeline_clone_and_params():
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    c = clone(pipe)
    assert c is not pipe
    params = pipe.get_params()
    assert "logisticregression__C" in params
    pipe.set_params(logisticregression__C=5.0)
    assert pipe.named_steps["logisticregression"].C == 5.0


def test_cross_val_score_on_pipeline():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=1)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=5)
    assert len(scores) == 5 and scores.mean() > 0.9


def test_feature_union():
    X = np.random.RandomState(0).normal(size=(50, 4))
    fu = FeatureUnion([("pca", PCA(n_components=2)),
                       ("scale", StandardScaler())])
    Xt = fu.fit_transform(X)
    assert Xt.shape == (50, 6)


def test_column_transformer():
    X = np.random.RandomState(1).normal(size=(30, 5))
    ct = ColumnTransformer([
        ("std", StandardScaler(), [0, 1]),
        ("mm", MinMaxScaler(), [2]),
    ], remainder="passthrough")
    Xt = ct.fit_transform(X)
    assert Xt.shape == (30, 5)
    assert np.allclose(Xt[:, 0].mean(), 0, atol=1e-10)
    assert Xt[:, 2].min() == 0 and Xt[:, 2].max() == 1
    assert np.allclose(Xt[:, 3:], X[:, 3:])
