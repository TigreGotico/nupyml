"""Time-series classification: ROCKET, shapelets, BOSS, DTW-kNN, features, MP.

Each method is tested on data suited to the property it claims -- ROCKET/DTW/
shapelets/features on a shape-at-random-position problem (the pattern's location
varies, so absolute columns are uninformative), BOSS on frequency-content classes
(its low-pass symbolic strength), and the matrix profile on a series with a
planted repeated motif and a planted discord. Adversarial cases check short
series, determinism, and numerosity reduction.
"""
import numpy as np
import pytest

from nupyml.tsclass import (Rocket, RocketClassifier, ShapeletTransform,
                           ShapeletTransformClassifier, BOSS,
                           KNeighborsTimeSeriesClassifier, TSFeatureExtractor,
                           FEATURE_NAMES, matrix_profile, motif, discord)


def _shape_dataset(seed=0, n_per=40, L=48):
    """3 classes sharing a random-phase carrier; they differ only by a small
    up-bump / down-notch / nothing at a RANDOM position."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for label in (0, 1, 2):
        for _ in range(n_per):
            t = np.linspace(0, 1, L)
            carrier = 0.5 * np.sin(2 * np.pi * 2 * t + rng.uniform(0, 2 * np.pi))
            noise = 0.2 * rng.randn(L)
            centre = rng.uniform(0.3, 0.7)
            local = np.exp(-((t - centre) ** 2) / (2 * 0.008))
            mark = {0: 0.0, 1: 1.5, 2: -1.5}[label] * local
            X.append(carrier + mark + noise)
            y.append(label)
    X, y = np.array(X), np.array(y)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    n_tr = int(0.7 * len(y))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def _freq_dataset(seed=1, n_per=50, L=96):
    """2 classes distinguished purely by frequency content -- BOSS's domain."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per):
            t = np.linspace(0, 1, L)
            freq = (2, 5)[label]
            X.append(np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                     + 0.3 * rng.randn(L))
            y.append(label)
    X, y = np.array(X), np.array(y)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    n_tr = int(0.7 * len(y))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


# --- ROCKET ---------------------------------------------------------------

def test_rocket_transform_shape_and_determinism():
    Xtr, ytr, Xte, yte = _shape_dataset()
    r = Rocket(n_kernels=100, random_state=0).fit(Xtr)
    F1 = r.transform(Xte)
    assert F1.shape == (len(Xte), 200)              # 2 features per kernel
    # ppv features live in [0, 1]
    ppv = F1[:, 0::2]
    assert ppv.min() >= 0.0 and ppv.max() <= 1.0
    # same seed -> identical kernels -> identical features
    F2 = Rocket(n_kernels=100, random_state=0).fit(Xtr).transform(Xte)
    assert np.allclose(F1, F2)


def test_rocket_classifier_solves_shape_task():
    Xtr, ytr, Xte, yte = _shape_dataset()
    clf = RocketClassifier(n_kernels=300, random_state=0).fit(Xtr, ytr)
    assert (clf.predict(Xte) == yte).mean() > 0.9


def test_rocket_handles_series_shorter_than_some_kernels():
    """A short series must not crash the dilated convolution."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 12)
    F = Rocket(n_kernels=50, random_state=0).fit(X).transform(X)
    assert F.shape == (10, 100) and np.all(np.isfinite(F))


# --- DTW k-NN -------------------------------------------------------------

def test_dtw_knn_solves_shape_task():
    Xtr, ytr, Xte, yte = _shape_dataset()
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, window=4).fit(Xtr, ytr)
    assert (clf.predict(Xte) == yte).mean() > 0.85


def test_dtw_beats_euclidean_under_phase_shift():
    """On phase-shifted patterns, warping should help vs a rigid Euclidean 1-NN."""
    Xtr, ytr, Xte, yte = _shape_dataset()
    dtw = KNeighborsTimeSeriesClassifier(metric="dtw", window=6).fit(Xtr, ytr)
    euc = KNeighborsTimeSeriesClassifier(metric="euclidean").fit(Xtr, ytr)
    assert (dtw.predict(Xte) == yte).mean() >= (euc.predict(Xte) == yte).mean()


# --- shapelets ------------------------------------------------------------

def test_shapelet_transform_shapes_and_nonnegative():
    Xtr, ytr, Xte, yte = _shape_dataset()
    st = ShapeletTransform(n_shapelets=30, min_length=4, max_length=12,
                           random_state=0).fit(Xtr)
    D = st.transform(Xte)
    assert D.shape == (len(Xte), 30)
    assert np.all(D >= 0)                            # distances are non-negative


def test_shapelet_classifier_beats_chance():
    Xtr, ytr, Xte, yte = _shape_dataset()
    clf = ShapeletTransformClassifier(n_shapelets=200, min_length=4,
                                      max_length=16, random_state=0).fit(Xtr, ytr)
    assert (clf.predict(Xte) == yte).mean() > 0.55   # chance is 1/3


# --- BOSS -----------------------------------------------------------------

def test_boss_classifies_frequency_content():
    Xtr, ytr, Xte, yte = _freq_dataset()
    clf = BOSS(word_length=4, alphabet_size=4).fit(Xtr, ytr)
    assert (clf.predict(Xte) == yte).mean() > 0.9


def test_boss_numerosity_reduction_collapses_constant_runs():
    """A constant series produces the SAME word at every window (each window
    z-normalises to zeros); numerosity reduction must count it once, not once per
    window."""
    Xtr, ytr, _, _ = _freq_dataset()
    clf = BOSS(word_length=4, alphabet_size=4).fit(Xtr, ytr)   # non-degenerate fit
    constant = np.full(Xtr.shape[1], 3.0)
    hist = clf._histogram(constant, clf.w_)
    n_windows = len(constant) - clf.w_ + 1
    assert n_windows > 5                              # there ARE many windows...
    assert sum(hist.values()) == 1                    # ...collapsed to one count


# --- feature bank ---------------------------------------------------------

def test_feature_extractor_shape_and_names():
    Xtr, ytr, Xte, yte = _shape_dataset()
    ext = TSFeatureExtractor().fit(Xtr)
    F = ext.transform(Xtr)
    assert F.shape == (len(Xtr), len(FEATURE_NAMES))
    assert list(ext.get_feature_names_out()) == FEATURE_NAMES
    assert np.all(np.isfinite(F))


def test_feature_pipeline_classifies():
    from nupyml.pipeline import make_pipeline
    from nupyml.ensemble import RandomForestClassifier
    Xtr, ytr, Xte, yte = _shape_dataset()
    model = make_pipeline(TSFeatureExtractor(),
                          RandomForestClassifier(n_estimators=100, random_state=0))
    model.fit(Xtr, ytr)
    assert (model.predict(Xte) == yte).mean() > 0.8


# --- matrix profile -------------------------------------------------------

def test_matrix_profile_finds_planted_motif():
    """Two copies of the same shape planted at known positions must be each
    other's nearest neighbour (the profile minimum)."""
    rng = np.random.RandomState(0)
    m = 20
    pattern = np.sin(np.linspace(0, 3, m))
    s = np.concatenate([rng.randn(30), pattern, rng.randn(15), pattern,
                        rng.randn(20)])
    i, j = motif(s, m)
    # the motif pair should be the two planted copies (starts 30 and 65)
    assert {i, j} == {30, 65}


def test_matrix_profile_discord_is_the_odd_one_out():
    """A single injected spike region should be the most anomalous subsequence."""
    rng = np.random.RandomState(0)
    m = 16
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 20)), 6)[:200]
    base[90:106] += 5.0                              # an anomaly
    d = discord(base, m)
    assert 80 <= d <= 106                            # near the injected region


def test_matrix_profile_rejects_too_short_series():
    with pytest.raises(ValueError):
        matrix_profile(np.arange(5.0), window=5)


def test_matrix_profile_length_and_finiteness():
    rng = np.random.RandomState(0)
    s = rng.randn(120)
    prof, idx = matrix_profile(s, 16)
    assert len(prof) == len(idx) == 120 - 16 + 1
    assert np.all(np.isfinite(prof)) and prof.min() >= 0
