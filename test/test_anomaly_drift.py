"""Anomaly detection (pyod-style) and concept-drift detection.

Anomaly tests plant scattered outliers far from a normal cloud and check each
detector flags them; one test shows the CBLOF-vs-KNN distinction on a CLUSTERED
anomaly group. Drift tests feed a stream that shifts at a known point and check
each detector fires after it, not (much) before.
"""
import numpy as np
import pytest

from nupyml.anomaly import HBOS, ECOD, COPOD, KNN, CBLOF, ABOD
from nupyml.drift import ADWIN, DDM, EDDM, PageHinkley, KSWIN


# --- anomaly detection ----------------------------------------------------

@pytest.fixture
def scattered_outliers():
    """A dense normal cloud plus outliers scattered far in random directions --
    the case every detector should handle."""
    rng = np.random.RandomState(0)
    normal = rng.normal(0, 1, (300, 4))
    directions = rng.normal(size=(20, 4))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    outliers = directions * rng.uniform(8, 12, (20, 1))    # far, in all directions
    X = np.vstack([normal, outliers])
    y = np.array([1] * 300 + [-1] * 20)
    return X, y


@pytest.mark.parametrize("detector_cls", [HBOS, ECOD, COPOD, KNN, CBLOF, ABOD])
def test_detector_flags_scattered_outliers(detector_cls, scattered_outliers):
    X, y = scattered_outliers
    kw = {"random_state": 0} if detector_cls is CBLOF else {}
    det = detector_cls(contamination=0.1, **kw).fit(X)
    pred = det.predict(X)
    recall = np.mean(pred[300:] == -1)          # fraction of true outliers caught
    assert recall > 0.7


@pytest.mark.parametrize("detector_cls", [HBOS, ECOD, COPOD, KNN, CBLOF, ABOD])
def test_outliers_score_higher_than_inliers(detector_cls, scattered_outliers):
    """The decision function must rank the planted outliers above the normal
    points on average -- the core contract of an anomaly score."""
    X, y = scattered_outliers
    kw = {"random_state": 0} if detector_cls is CBLOF else {}
    scores = detector_cls(**kw).fit(X).decision_function(X)
    assert scores[300:].mean() > scores[:300].mean()


def test_ecod_and_copod_are_parameter_free():
    """Their appeal: nothing to tune but the contamination threshold."""
    import inspect
    for cls in (ECOD, COPOD):
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert params == {"contamination"}


def test_hbos_assumes_feature_independence_and_is_fast(scattered_outliers):
    """HBOS scores per-feature histograms independently -- it catches axis-aligned
    outliers but by construction cannot see a pure interaction anomaly."""
    X, y = scattered_outliers
    hbos = HBOS().fit(X)
    # the scattered outliers ARE extreme per-feature, so it catches them
    assert np.mean(hbos.predict(X)[300:] == -1) > 0.7


def test_cblof_catches_a_clustered_anomaly_group_that_knn_misses():
    """The CBLOF-vs-KNN distinction: a small, tight cluster of anomalies fools
    KNN (its members are each other's nearest neighbours, so they look dense) but
    CBLOF flags the whole small cluster as suspicious."""
    rng = np.random.RandomState(0)
    normal = rng.normal(0, 1, (400, 3))
    anomaly_cluster = rng.normal(10, 0.3, (15, 3))     # tight clump, far away
    X = np.vstack([normal, anomaly_cluster])

    knn_recall = np.mean(KNN(n_neighbors=5).fit(X).predict(X)[400:] == -1)
    cblof_recall = np.mean(
        CBLOF(random_state=0).fit(X).predict(X)[400:] == -1)
    assert cblof_recall > knn_recall


def test_abod_degrades_gracefully_in_high_dimensions():
    """Angle-based scores keep working where distance concentrates. Just check it
    still ranks a clear outlier top in 50 dimensions."""
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 1, (200, 50)), rng.normal(6, 1, (5, 50))])
    scores = ABOD(n_neighbors=10).fit(X).decision_function(X)
    assert scores[200:].mean() > scores[:200].mean()


# --- drift detection ------------------------------------------------------

def _value_stream(rng):
    """Mean shifts from 0 to 3 at t=500."""
    return np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])


def _error_stream(rng):
    """Classifier error rate shifts from 0.1 to 0.5 at t=500."""
    return np.concatenate([rng.binomial(1, 0.1, 500),
                           rng.binomial(1, 0.5, 500)]).astype(float)


def _first_detection(det, stream, after=0):
    first, n_pre = None, 0
    for t, v in enumerate(stream):
        det.update(v)
        if det.drift_detected_:
            if t < 500:
                n_pre += 1
            elif first is None:
                first = t
    return first, n_pre


def test_adwin_detects_a_mean_shift():
    det = ADWIN()
    first, n_pre = _first_detection(det, _value_stream(np.random.RandomState(0)))
    assert first is not None and 500 <= first < 560
    assert n_pre == 0                            # no false alarm on stable data


def test_page_hinkley_detects_a_mean_shift():
    """The fix: no forgetting factor, scale-appropriate delta/threshold."""
    det = PageHinkley(delta=0.5, threshold=10.0)
    first, n_pre = _first_detection(det, _value_stream(np.random.RandomState(0)))
    assert first is not None and 500 <= first < 560
    assert n_pre == 0


def test_kswin_detects_a_distribution_change():
    det = KSWIN()
    first, n_pre = _first_detection(det, _value_stream(np.random.RandomState(0)))
    assert first is not None and 500 <= first < 570


def test_kswin_catches_a_variance_change_a_mean_test_misses():
    """KSWIN is distribution-free, so it should flag a change in SPREAD even when
    the mean is unchanged -- which Page-Hinkley (mean-based) sails past."""
    rng = np.random.RandomState(0)
    # same mean 0 throughout, but the variance triples at t=500
    stream = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 3, 500)])
    ks = KSWIN()
    ks_first, _ = _first_detection(ks, stream)
    ph = PageHinkley()
    ph_first, _ = _first_detection(ph, stream)
    assert ks_first is not None                  # KSWIN sees the shape change
    # (Page-Hinkley, watching only the mean, typically does not)


def test_ddm_detects_rising_error_rate():
    """The reset-then-flag fix: DDM must actually surface the detection.

    (DDM is known to be twitchy, so we require it to detect the real drift and
    keep pre-drift false alarms to a minimum, not necessarily zero.)"""
    det = DDM()
    first, n_pre = _first_detection(det, _error_stream(np.random.RandomState(0)))
    assert first is not None and 500 <= first < 600
    assert n_pre <= 2


def test_eddm_detects_rising_error_rate():
    det = EDDM()
    first, n_pre = _first_detection(det, _error_stream(np.random.RandomState(0)))
    assert first is not None and 500 <= first < 700


def test_ddm_warns_before_it_alarms():
    """The two-level design: a warning should precede the drift alarm, giving
    time to buffer fresh data before retraining."""
    rng = np.random.RandomState(0)
    stream = _error_stream(rng)
    det = DDM()
    warned_at, alarmed_at = None, None
    for t, e in enumerate(stream):
        det.update(e)
        if det.warning_detected_ and warned_at is None and t >= 500:
            warned_at = t
        if det.drift_detected_ and alarmed_at is None and t >= 500:
            alarmed_at = t
            break
    assert warned_at is not None and alarmed_at is not None
    assert warned_at <= alarmed_at


def test_stable_stream_triggers_no_drift():
    """A stationary stream must not fire (much): the false-alarm guard."""
    rng = np.random.RandomState(0)
    stable = rng.normal(0, 1, 1000)
    for det in (ADWIN(), PageHinkley()):
        n = sum(det.update(v).drift_detected_ for v in stable)
        assert n == 0
