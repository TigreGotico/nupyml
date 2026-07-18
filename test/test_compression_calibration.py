"""G11: model compression and probability calibration.

Compression: pruning hits the target sparsity, quantization stays close, low-rank
reconstructs a low-rank matrix, distillation transfers the teacher's predictions.
Calibration: ECE is zero when calibrated and positive when over-confident;
temperature scaling lowers ECE without changing accuracy; the binary calibrators
reduce ECE.
"""
import numpy as np
import pytest

from nupyml import nn
from nupyml.nn import (magnitude_prune, quantize_weights, low_rank_approximation,
                      KnowledgeDistillation, Sequential, Linear, ReLU)
from nupyml.calibration import (expected_calibration_error, TemperatureScaling,
                               HistogramBinning, BetaCalibration)
from nupyml.datasets import make_classification
from nupyml.autograd import Tensor


# --- compression ----------------------------------------------------------

def test_magnitude_prune_hits_target_sparsity():
    net = Sequential(Linear(20, 32), ReLU(), Linear(32, 10))
    frac = magnitude_prune(net, sparsity=0.6)
    assert abs(frac - 0.6) < 0.05
    # the weight matrices really contain that fraction of zeros
    zeros = sum((p.data == 0).sum() for p in net.parameters() if p.data.ndim >= 2)
    total = sum(p.data.size for p in net.parameters() if p.data.ndim >= 2)
    assert abs(zeros / total - 0.6) < 0.05


def test_quantize_weights_stays_close_and_discrete():
    net = Sequential(Linear(20, 32), ReLU(), Linear(32, 10))
    before = [p.data.copy() for p in net.parameters()]
    err = quantize_weights(net, bits=8)
    assert err < 0.01                                 # 8-bit error is small
    # each tensor now takes few distinct values
    first = list(net.parameters())[0].data
    assert len(np.unique(first)) <= 256


def test_low_rank_approximation_reconstructs_low_rank_matrix():
    rng = np.random.RandomState(0)
    # a genuinely rank-5 matrix
    W = rng.randn(30, 5) @ rng.randn(5, 20)
    A, B = low_rank_approximation(W, rank=5)
    assert A.shape == (30, 5) and B.shape == (5, 20)
    assert np.linalg.norm(W - A @ B) / np.linalg.norm(W) < 1e-6
    # a lower rank than the true rank loses information
    A3, B3 = low_rank_approximation(W, rank=3)
    assert np.linalg.norm(W - A3 @ B3) > 1e-3


def test_knowledge_distillation_transfers_teacher_predictions():
    X, y = make_classification(n_samples=500, n_features=12, n_informative=6,
                               n_classes=3, random_state=0)
    # teacher: a wide net trained on the labels
    teacher = Sequential(Linear(12, 64), ReLU(), Linear(64, 3))
    opt = nn.Adam(teacher.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    for _ in range(150):
        opt.zero_grad(); loss(teacher(Tensor(X)), y).backward(); opt.step()
    teacher_pred = teacher(Tensor(X)).data.argmax(1)
    # student: a small net distilled from the teacher
    student = Sequential(Linear(12, 8), ReLU(), Linear(8, 3))
    kd = KnowledgeDistillation(student, teacher, temperature=4.0, alpha=0.5,
                               epochs=80, random_state=0).fit(X, y)
    agree = np.mean(kd.predict(X) == teacher_pred)
    assert agree > 0.8                                # student mimics the teacher


# --- calibration ----------------------------------------------------------

def test_ece_zero_when_calibrated_positive_when_overconfident():
    rng = np.random.RandomState(0)
    n = 2000
    # perfectly calibrated binary: label ~ Bernoulli(p) with p the stated conf
    p = rng.uniform(0, 1, n)
    y = (rng.rand(n) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.05
    # over-confident: push probs toward 0/1
    over = np.clip((p - 0.5) * 3 + 0.5, 0, 1)
    assert expected_calibration_error(y, over) > expected_calibration_error(y, p)


def test_temperature_scaling_lowers_ece_without_changing_accuracy():
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                               n_classes=3, random_state=0)
    from nupyml.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(X[:400], y[:400])
    logits = clf.decision_function(X[400:]) * 3.0     # artificially over-confident
    raw = np.exp(logits - logits.max(1, keepdims=True))
    raw /= raw.sum(1, keepdims=True)
    ts = TemperatureScaling().fit(logits, y[400:])
    cal = ts.transform(logits)
    assert ts.temperature_ > 1.0                      # softening an over-confident model
    assert expected_calibration_error(y[400:], cal) <= \
        expected_calibration_error(y[400:], raw) + 1e-9
    # accuracy (argmax) is unchanged by a positive temperature
    assert np.all(raw.argmax(1) == cal.argmax(1))


@pytest.mark.parametrize("Cal", [HistogramBinning, BetaCalibration])
def test_binary_calibrator_reduces_ece(Cal):
    rng = np.random.RandomState(0)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    y = (rng.rand(n) < true_p).astype(int)
    # a miscalibrated score: a squashed version of the true probability
    scores = true_p ** 2
    cal = Cal().fit(scores[:1000], y[:1000])
    calibrated = cal.transform(scores[1000:])
    assert expected_calibration_error(y[1000:], calibrated) < \
        expected_calibration_error(y[1000:], scores[1000:])
