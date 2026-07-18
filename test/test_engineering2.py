"""I8: feature engineering v2 -- supervised MDLP binning and datetime features.

MDLP places a cut only where it reduces class entropy enough to pay for itself:
a perfectly separable feature gets exactly one edge at the boundary; a feature
independent of the target gets none. DateTimeFeatures unpacks a timestamp into
correct calendar fields plus their cyclical encodings.
"""
import numpy as np
import pytest

from nupyml.encoders import MDLPDiscretizer, DateTimeFeatures


def test_mdlp_finds_the_single_separating_cut():
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.uniform(0, 5, 100),
                        rng.uniform(5, 10, 100)]).reshape(-1, 1)
    y = np.array([0] * 100 + [1] * 100)
    m = MDLPDiscretizer().fit(x, y)
    assert len(m.edges_[0]) == 1                      # one cut, not many
    assert abs(m.edges_[0][0] - 5.0) < 0.2            # at the true boundary
    # samples land in the class-pure bins
    bins = m.transform(np.array([[2.0], [8.0]])).ravel()
    assert bins[0] == 0 and bins[1] == 1


def test_mdlp_makes_no_cut_on_an_irrelevant_feature():
    rng = np.random.RandomState(1)
    x = rng.randn(200, 1)                             # independent of y
    y = np.array([0, 1] * 100)
    m = MDLPDiscretizer().fit(x, y)
    assert len(m.edges_[0]) == 0                      # MDL rejects every cut
    assert np.all(m.transform(x) == 0)               # single bin


def test_mdlp_multiclass_staircase():
    # three well-separated blocks -> two cuts, three bins
    x = np.concatenate([np.linspace(0, 1, 60), np.linspace(3, 4, 60),
                        np.linspace(6, 7, 60)]).reshape(-1, 1)
    y = np.array([0] * 60 + [1] * 60 + [2] * 60)
    m = MDLPDiscretizer().fit(x, y)
    assert len(m.edges_[0]) == 2
    assert int(m.transform([[0.5]])[0, 0]) == 0
    assert int(m.transform([[6.5]])[0, 0]) == 2


def test_datetime_features_are_correct():
    dts = np.array(['2021-01-01T09:30', '2021-07-04T18:00', '2021-12-25T00:00'],
                   dtype='datetime64[s]')
    dt = DateTimeFeatures(cyclical=True)
    F = dt.transform(dts)
    idx = {n: i for i, n in enumerate(dt.feature_names_)}
    # 2021-01-01 is a Friday (Monday=0 -> 4) at 09:00
    assert F[0, idx["year"]] == 2021
    assert F[0, idx["month"]] == 1 and F[0, idx["day"]] == 1
    assert F[0, idx["dayofweek"]] == 4 and F[0, idx["hour"]] == 9
    assert F[0, idx["is_weekend"]] == 0
    # Jul 4 (Sun) and Dec 25 (Sat) 2021 are weekends
    assert F[1, idx["is_weekend"]] == 1 and F[2, idx["is_weekend"]] == 1
    assert F[1, idx["quarter"]] == 3 and F[2, idx["quarter"]] == 4


def test_datetime_cyclical_wraps_december_to_january():
    dts = np.array(['2021-01-15', '2021-12-15'], dtype='datetime64[s]')
    dt = DateTimeFeatures(cyclical=True)
    F = dt.transform(dts)
    idx = {n: i for i, n in enumerate(dt.feature_names_)}
    jan = np.array([F[0, idx["month_sin"]], F[0, idx["month_cos"]]])
    dec = np.array([F[1, idx["month_sin"]], F[1, idx["month_cos"]]])
    # adjacent months on the circle are closer than opposite months would be
    assert np.linalg.norm(jan - dec) < 1.0


def test_datetime_accepts_unix_seconds():
    # 1609495200 == 2021-01-01T10:00:00 UTC
    F = DateTimeFeatures(cyclical=False).transform(np.array([1609495200]))
    assert F.shape[0] == 1
