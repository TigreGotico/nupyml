"""I12: meta-learning -- prototypical / matching nets, Reptile, MAML, meta-features.

The few-shot nets must classify UNSEEN classes from a handful of shots. The
gradient meta-learners must produce an initialisation that, after a few adaptation
steps on a new task, beats a random initialisation adapted the same way (averaged
over many tasks). Meta-features must report correct dataset descriptors.
"""
import numpy as np
import pytest

from nupyml.meta import (PrototypicalNetwork, MatchingNetwork, Reptile, MAML,
                         extract_meta_features)
from nupyml.meta._gradient import _MLP


def _fewshot_data(rng):
    centers = rng.randn(6, 5) * 6
    X = np.vstack([c + rng.randn(30, 5) for c in centers])
    y = np.repeat(np.arange(6), 30)
    return X, y


def _episode(X, y, classes):
    sx, sy, qx, qy = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        sx.append(X[idx[:5]]); sy += [c] * 5
        qx.append(X[idx[5:15]]); qy += [c] * 10
    return np.vstack(sx), np.array(sy), np.vstack(qx), np.array(qy)


def test_prototypical_network_classifies_unseen_classes():
    rng = np.random.RandomState(0)
    X, y = _fewshot_data(rng)
    proto = PrototypicalNetwork(n_way=3, k_shot=5, n_query=5, episodes=300,
                                random_state=0).fit(X[y < 4], y[y < 4])
    sx, sy, qx, qy = _episode(X, y, [3, 4, 5])           # classes 4,5 never trained on
    acc = (proto.predict(sx, sy, qx) == qy).mean()
    assert acc > 0.9


def test_matching_network_classifies_unseen_classes():
    rng = np.random.RandomState(0)
    X, y = _fewshot_data(rng)
    match = MatchingNetwork(n_way=3, k_shot=5, n_query=5, episodes=300,
                            random_state=0).fit(X[y < 4], y[y < 4])
    sx, sy, qx, qy = _episode(X, y, [3, 4, 5])
    acc = (match.predict(sx, sy, qx) == qy).mean()
    assert acc > 0.9


def _sine_sampler(rng):
    def sampler():
        amp = rng.uniform(0.5, 2.0); ph = rng.uniform(0, np.pi)
        xt = rng.uniform(-5, 5, 20)
        return xt, amp * np.sin(xt + ph)
    return sampler


def _adapt_vs_random(meta, n_tasks=20):
    tr = np.random.RandomState(42)
    meta_mse, rnd_mse = [], []
    for _ in range(n_tasks):
        amp = tr.uniform(0.5, 2.0); ph = tr.uniform(0, np.pi)
        xt = np.linspace(-5, 5, 10); yt = amp * np.sin(xt + ph)
        xtest = np.linspace(-5, 5, 50); ytest = amp * np.sin(xtest + ph)
        meta.adapt(xt, yt, steps=5)
        meta_mse.append(np.mean((meta.predict(xtest) - ytest) ** 2))
        m = _MLP(1, meta.hidden, np.random.RandomState(7))
        m.sgd(xt.reshape(-1, 1), yt, 0.02, 5)
        rnd_mse.append(np.mean((m.forward(xtest.reshape(-1, 1))[0].ravel()
                                - ytest) ** 2))
    return np.mean(meta_mse), np.mean(rnd_mse)


def test_reptile_initialisation_beats_random():
    rng = np.random.RandomState(0)
    rep = Reptile(hidden=40, inner_lr=0.02, inner_steps=5, meta_lr=0.1,
                  random_state=0).meta_train(_sine_sampler(rng), n_iterations=1000)
    meta_mse, rnd_mse = _adapt_vs_random(rep)
    assert meta_mse < rnd_mse


def test_maml_initialisation_beats_random():
    rng = np.random.RandomState(0)
    maml = MAML(hidden=40, inner_lr=0.02, inner_steps=5, meta_lr=0.05,
                random_state=0).meta_train(_sine_sampler(rng), n_iterations=1000)
    meta_mse, rnd_mse = _adapt_vs_random(maml)
    assert meta_mse < rnd_mse


def test_meta_features_describe_the_dataset():
    rng = np.random.RandomState(0)
    X, y = _fewshot_data(rng)
    mf = extract_meta_features(X, y)
    assert mf["n_samples"] == 180 and mf["n_features"] == 5
    assert mf["n_classes"] == 6
    assert mf["imbalance_ratio"] == pytest.approx(1.0)   # 30 per class
    assert mf["landmark_1nn"] > 0.9                       # well-separated blobs
    # class_entropy of 6 balanced classes is log2(6)
    assert mf["class_entropy"] == pytest.approx(np.log2(6), abs=1e-6)


def test_meta_features_detect_imbalance():
    rng = np.random.RandomState(0)
    X = rng.randn(110, 3)
    y = np.array([0] * 100 + [1] * 10)                    # 10:1 imbalance
    mf = extract_meta_features(X, y)
    assert mf["imbalance_ratio"] == pytest.approx(10.0)
