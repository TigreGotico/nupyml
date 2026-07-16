import numpy as np
import pytest

from nupyml.autograd import Tensor, no_grad, gradcheck
from nupyml.autograd import functional as F

rng = np.random.RandomState(42)


def t(*shape):
    return Tensor(rng.normal(size=shape), requires_grad=True)


# ---------------------------------------------------------------------------
# elementwise + broadcasting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    lambda a, b: a / (b + 3.0),
])
def test_binary_ops(fn):
    assert gradcheck(fn, [t(3, 4), t(3, 4)])


def test_broadcasting_grads():
    assert gradcheck(lambda a, b: a * b, [t(3, 4), t(4)])
    assert gradcheck(lambda a, b: a + b, [t(2, 1, 4), t(3, 1)])
    assert gradcheck(lambda a, b: a / (b.abs() + 1.0), [t(5, 1), t(1, 5)])


@pytest.mark.parametrize("fn", [
    lambda a: a.exp(),
    lambda a: (a.abs() + 1.0).log(),
    lambda a: a.tanh(),
    lambda a: a.sigmoid(),
    lambda a: a ** 3,
    lambda a: (a.abs() + 0.5).sqrt(),
    lambda a: -a,
])
def test_unary_ops(fn):
    assert gradcheck(fn, [t(4, 3)])


def test_relu_and_leaky():
    x = Tensor(np.array([[-2.0, -0.5, 0.5, 2.0]]), requires_grad=True)
    assert gradcheck(lambda a: a.relu(), [x])
    assert gradcheck(lambda a: F.leaky_relu(a, 0.1), [x])
    assert gradcheck(lambda a: F.gelu(a), [t(3, 3)])


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------

def test_matmul_2d():
    assert gradcheck(lambda a, b: a @ b, [t(3, 4), t(4, 5)])


def test_matmul_vec():
    assert gradcheck(lambda a, b: a @ b, [t(3, 4), t(4)])
    assert gradcheck(lambda a, b: a @ b, [t(4), t(4, 3)])


def test_matmul_batched():
    assert gradcheck(lambda a, b: a @ b, [t(2, 3, 4), t(2, 4, 5)])


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda a: a.sum(),
    lambda a: a.sum(axis=0),
    lambda a: a.sum(axis=1, keepdims=True),
    lambda a: a.mean(),
    lambda a: a.mean(axis=1),
    lambda a: a.var(axis=0),
])
def test_reductions(fn):
    assert gradcheck(fn, [t(3, 5)])


def test_max_reduction():
    x = Tensor(np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]]), requires_grad=True)
    assert gradcheck(lambda a: a.max(), [x])
    assert gradcheck(lambda a: a.max(axis=1), [x])
    assert gradcheck(lambda a: a.min(axis=0), [x])


# ---------------------------------------------------------------------------
# shape ops
# ---------------------------------------------------------------------------

def test_shape_ops():
    assert gradcheck(lambda a: a.reshape(6, 2), [t(3, 4)])
    assert gradcheck(lambda a: a.transpose(), [t(3, 4)])
    assert gradcheck(lambda a: a.transpose(1, 0, 2), [t(2, 3, 4)])
    assert gradcheck(lambda a: a[1:, ::2], [t(4, 6)])
    assert gradcheck(lambda a: a.pad(((1, 1), (2, 0))), [t(3, 3)])


def test_concat_stack_where():
    assert gradcheck(lambda a, b: Tensor.concatenate([a, b], axis=1), [t(2, 3), t(2, 4)])
    assert gradcheck(lambda a, b: Tensor.stack([a, b], axis=0), [t(3, 3), t(3, 3)])
    cond = rng.uniform(size=(3, 3)) > 0.5
    assert gradcheck(lambda a, b: Tensor.where(cond, a, b), [t(3, 3), t(3, 3)])


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------

def test_softmax():
    assert gradcheck(lambda a: F.softmax(a, axis=-1), [t(4, 5)])
    assert gradcheck(lambda a: F.log_softmax(a, axis=-1), [t(4, 5)])
    s = F.softmax(t(3, 7)).data
    assert np.allclose(s.sum(axis=-1), 1)


# ---------------------------------------------------------------------------
# conv / pool / embedding
# ---------------------------------------------------------------------------

def test_conv2d():
    x = t(2, 3, 6, 6)
    w = t(4, 3, 3, 3)
    b = t(4)
    assert gradcheck(lambda a, ww, bb: F.conv2d(a, ww, bb, stride=1, padding=1),
                     [x, w, b], eps=1e-5, rtol=1e-3, atol=1e-5)
    out = F.conv2d(x, w, b, stride=2, padding=0)
    assert out.shape == (2, 4, 2, 2)


def test_conv2d_matches_scipy():
    from scipy.signal import correlate2d
    x = rng.normal(size=(1, 1, 8, 8))
    w = rng.normal(size=(1, 1, 3, 3))
    out = F.conv2d(Tensor(x), Tensor(w)).data[0, 0]
    ref = correlate2d(x[0, 0], w[0, 0], mode="valid")
    assert np.allclose(out, ref)


def test_pooling():
    x = t(2, 3, 6, 6)
    assert gradcheck(lambda a: F.max_pool2d(a, 2), [x], eps=1e-5, rtol=1e-3)
    assert gradcheck(lambda a: F.avg_pool2d(a, 2), [x])
    assert F.max_pool2d(x, 2).shape == (2, 3, 3, 3)


def test_embedding():
    w = t(10, 4)
    idx = np.array([[1, 2], [2, 7]])
    assert gradcheck(lambda ww: F.embedding(idx, ww), [w])
    assert F.embedding(idx, w).shape == (2, 2, 4)


def test_dropout():
    x = t(100, 100)
    rng2 = np.random.RandomState(0)
    out = F.dropout(x, p=0.5, training=True, rng=rng2)
    assert abs((out.data == 0).mean() - 0.5) < 0.05
    out_eval = F.dropout(x, p=0.5, training=False)
    assert out_eval is x


# ---------------------------------------------------------------------------
# machinery
# ---------------------------------------------------------------------------

def test_no_grad():
    x = t(3, 3)
    with no_grad():
        y = x * 2
    assert not y.requires_grad


def test_grad_accumulation_diamond():
    # y = x*x + x*x -> dy/dx = 4x
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    a = x * x
    b = x * x
    (a + b).sum().backward()
    assert np.allclose(x.grad, 4 * x.data)


def test_reused_node():
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = x * 2
    z = y * y + y  # dz/dy = 2y+1 = 13; dz/dx = 26
    z.backward()
    assert np.allclose(x.grad, [26.0])


def test_detach_stops_grad():
    x = t(3)
    y = x.detach() * 2
    assert not y.requires_grad
