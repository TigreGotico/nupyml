"""G2: additional optimizers, weight-averaging wrappers, and LR schedules.

Optimizers must drive a convex quadratic to its minimum; SAM must run its two-step
update; SWA/EMA must track a running average of the weights; and each schedule
must produce the LR curve it advertises.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.nn import (Parameter, SGD, Adadelta, AMSGrad, LAMB, SAM, SWA, EMA,
                      ExponentialLR, PolynomialLR, CyclicalLR, OneCycleLR)


def _minimise(make_opt, steps=1200):
    """Minimise f(w) = sum((w - target)^2); return the final distance to target."""
    target = np.array([3.0, -2.0, 1.0])
    w = Parameter(np.zeros(3))
    opt = make_opt([w])
    for _ in range(steps):
        opt.zero_grad()
        loss = ((w - Tensor(target)) ** 2).sum()
        loss.backward()
        opt.step()
    return np.linalg.norm(w.data - target)


@pytest.mark.parametrize("make_opt", [
    lambda p: Adadelta(p),          # LR-free, slow to warm up but converges
    lambda p: AMSGrad(p, lr=0.1),
], ids=["adadelta", "amsgrad"])
def test_optimizer_reaches_the_minimum(make_opt):
    assert _minimise(make_opt) < 0.1


def test_lamb_fits_linear_regression():
    """LAMB is a large-model/large-batch optimizer; test it on its home turf --
    a linear regression where the layer-wise trust ratio behaves sensibly."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 10)
    w_true = rng.randn(10)
    y = X @ w_true
    Xt, yt = Tensor(X), Tensor(y)
    w = Parameter(np.zeros(10))
    opt = LAMB([w], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        (((Xt @ w - yt) ** 2).mean()).backward()
        opt.step()
    assert np.mean((X @ w.data - y) ** 2) < 0.1


def test_sam_two_step_decreases_loss():
    target = np.array([2.0, -1.0])
    w = Parameter(np.zeros(2))
    opt = SAM(SGD([w], lr=0.1), rho=0.05)

    def loss():
        return ((w - Tensor(target)) ** 2).sum()

    first = float(loss().data)
    for _ in range(200):
        opt.zero_grad(); loss().backward(); opt.first_step()
        opt.zero_grad(); loss().backward(); opt.second_step()
    assert float(loss().data) < first
    assert np.linalg.norm(w.data - target) < 0.2


def test_swa_averages_the_trajectory():
    w = Parameter(np.zeros(2))
    swa = SWA([w])
    positions = [np.array([1.0, 1.0]), np.array([3.0, 3.0]), np.array([2.0, 2.0])]
    for pos in positions:
        w.data[...] = pos
        swa.update_parameters()
    swa.swap_in()
    assert np.allclose(w.data, np.mean(positions, axis=0))   # mean of the iterates


def test_ema_tracks_a_decayed_average():
    w = Parameter(np.array([0.0]))
    ema = EMA([w], decay=0.9)
    for _ in range(100):
        w.data[...] = np.array([10.0])               # weights sit at 10
        ema.update()
    ema.copy_to()
    assert 9.0 < w.data[0] <= 10.0                   # shadow converges toward 10


# --- schedules ------------------------------------------------------------

class _DummyOpt:
    def __init__(self, lr):
        self.lr = lr


def _lr_curve(sched, n):
    lrs = [sched.optimizer.lr]
    for _ in range(n):
        sched.step()
        lrs.append(sched.optimizer.lr)
    return np.array(lrs)


def test_exponential_lr_decays_geometrically():
    opt = _DummyOpt(1.0)
    lrs = _lr_curve(ExponentialLR(opt, gamma=0.9), 5)
    assert lrs[1] == pytest.approx(0.9)
    assert lrs[2] == pytest.approx(0.81)


def test_polynomial_lr_hits_zero_at_the_end():
    opt = _DummyOpt(1.0)
    lrs = _lr_curve(PolynomialLR(opt, total_iters=10, power=1.0), 10)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(lrs) <= 1e-9)              # monotone non-increasing


def test_cyclical_lr_oscillates():
    opt = _DummyOpt(0.1)
    lrs = _lr_curve(CyclicalLR(opt, max_lr=1.0, step_size=5), 20)
    assert lrs.max() > 0.9 and lrs.min() < 0.2       # sweeps the full band
    # it goes up AND comes back down (not monotone)
    assert np.any(np.diff(lrs) > 0) and np.any(np.diff(lrs) < 0)


def test_one_cycle_warms_up_then_anneals():
    opt = _DummyOpt(0.01)
    lrs = _lr_curve(OneCycleLR(opt, max_lr=1.0, total_steps=100, pct_start=0.3), 100)
    peak = lrs.argmax()
    assert lrs[peak] == pytest.approx(1.0, abs=1e-6)
    assert 25 <= peak <= 35                           # peak near 30% of training
    assert lrs[-1] < 0.05                             # anneals to ~0
