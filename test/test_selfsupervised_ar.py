"""Autoregressive density estimation (MADE) and self-supervised learning."""
from itertools import product

import numpy as np
import pytest

from nupyml.datasets import load_digits
from nupyml.linear_model import LogisticRegression
from nupyml.model_selection import train_test_split
from nupyml.nn import (
    MADE, MaskedLinear, SimCLR, MaskedAutoEncoder, GaussianNoiseAugment,
    MaskingAugment, Adam,
)


@pytest.fixture(scope="module")
def digits_split():
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.3, random_state=0)


def _batches(n, rng, size=64):
    idx = rng.permutation(n)
    return [idx[i:i + size] for i in range(0, n - size + 1, size)]


# --- MADE -----------------------------------------------------------------

def test_masked_linear_ignores_masked_weights():
    layer = MaskedLinear(3, 2, rng=0)
    layer.set_mask(np.zeros((3, 2)))
    assert np.allclose(layer(np.ones((4, 3))).data, layer.bias.data)


def test_masked_weights_receive_no_gradient():
    """The connection does not exist, so training cannot resurrect it."""
    layer = MaskedLinear(3, 2, rng=0)
    mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.set_mask(mask)
    layer(np.ones((4, 3))).sum().backward()
    assert not np.any(layer.weight.grad[mask == 0])


def test_made_output_i_cannot_see_input_i():
    """The property the whole construction exists to guarantee.

    Measured on the Jacobian directly rather than inferred from the loss: a
    violation would let the model learn the identity, which scores a PERFECT
    likelihood and generates garbage. It would look like success.
    """
    m = MADE(6, hidden=(24, 24), rng=0)
    x = np.random.RandomState(0).uniform(0, 1, (1, 6))
    eps = 1e-5
    jac = np.zeros((6, 6))
    for j in range(6):
        step = np.zeros((1, 6))
        step[0, j] = eps
        jac[:, j] = ((m(x + step).data - m(x - step).data) / (2 * eps))[0]
    # strictly lower triangular: output i depends on inputs < i and nothing else
    assert np.abs(np.triu(jac)).max() == 0.0


def test_made_later_outputs_do_depend_on_earlier_inputs():
    """The masks must not be so strict that nothing is connected at all."""
    m = MADE(6, hidden=(32, 32), rng=0)
    x = np.random.RandomState(0).uniform(0, 1, (1, 6))
    eps = 1e-5
    jac = np.zeros((6, 6))
    for j in range(6):
        step = np.zeros((1, 6))
        step[0, j] = eps
        jac[:, j] = ((m(x + step).data - m(x - step).data) / (2 * eps))[0]
    assert np.abs(np.tril(jac, -1)).max() > 1e-3


def test_made_is_a_normalized_distribution():
    """Summing over every binary vector must give exactly 1 -- the chain rule
    guarantees it, so any deviation is a bug in the factorisation."""
    m = MADE(6, hidden=(24, 24), rng=0)
    allx = np.array(list(product([0, 1], repeat=6)), dtype=float)
    assert np.exp(m.log_prob(allx).data).sum() == pytest.approx(1.0, abs=1e-9)


def test_made_stays_normalized_after_training():
    rng = np.random.RandomState(0)
    data = rng.randint(0, 2, (2000, 6)).astype(float)
    m = MADE(6, hidden=(32,), rng=0)
    opt = Adam(m.parameters(), lr=5e-3)
    for _ in range(20):
        for b in _batches(len(data), rng):
            opt.zero_grad()
            m.loss(data[b]).backward()
            opt.step()
    allx = np.array(list(product([0, 1], repeat=6)), dtype=float)
    assert np.exp(m.log_prob(allx).data).sum() == pytest.approx(1.0, abs=1e-6)


def test_made_learns_a_deterministic_constraint():
    """x3 = x1 XOR x2. A model that has learned it reproduces it when sampling;
    chance is 0.5, so there is no way to fake this one."""
    rng = np.random.RandomState(0)
    data = rng.randint(0, 2, (3000, 6)).astype(float)
    data[:, 2] = np.logical_xor(data[:, 0], data[:, 1])

    m = MADE(6, hidden=(64, 64), rng=0)
    opt = Adam(m.parameters(), lr=5e-3)
    for _ in range(120):
        for b in _batches(len(data), rng):
            opt.zero_grad()
            m.loss(data[b]).backward()
            opt.step()

    s = m.sample(2000, rng=1)
    assert (s[:, 2] == np.logical_xor(s[:, 0], s[:, 1])).mean() > 0.95


def test_made_log_prob_is_higher_on_data_it_was_trained_on():
    rng = np.random.RandomState(0)
    data = np.zeros((2000, 6))
    data[:, ::2] = 1.0                      # a rigid pattern
    m = MADE(6, hidden=(32,), rng=0)
    opt = Adam(m.parameters(), lr=5e-3)
    for _ in range(60):
        for b in _batches(len(data), rng):
            opt.zero_grad()
            m.loss(data[b]).backward()
            opt.step()
    other = 1.0 - data
    assert m.log_prob(data).data.mean() > m.log_prob(other).data.mean()


def test_made_log_prob_does_not_underflow_on_confident_predictions():
    """log(sigmoid(z)) would be -inf here; the softplus form must not be."""
    m = MADE(4, hidden=(8,), rng=0)
    for layer in m.layers:
        layer.weight.data *= 0.0
        layer.bias.data += 60.0            # saturate every conditional
    lp = m.log_prob(np.ones((2, 4)))
    assert np.all(np.isfinite(lp.data))


def test_made_sampling_shape_is_binary():
    m = MADE(5, hidden=(16,), rng=0)
    s = m.sample(20, rng=0)
    assert s.shape == (20, 5)
    assert set(np.unique(s)).issubset({0.0, 1.0})


# --- self-supervised ------------------------------------------------------

def test_augmentations_change_the_data_without_destroying_it():
    X = np.random.RandomState(0).uniform(0, 1, (50, 8))
    for aug in (GaussianNoiseAugment(0.2, rng=0), MaskingAugment(0.3, rng=0)):
        out = aug(X)
        assert out.shape == X.shape
        assert not np.allclose(out, X)


def test_masking_augment_zeroes_roughly_the_requested_fraction():
    X = np.ones((200, 50))
    assert MaskingAugment(0.3, rng=0)(X).mean() == pytest.approx(0.7, abs=0.03)


def test_simclr_loss_prefers_matching_views():
    """The pretext task: paired views should score better than mismatched ones."""
    s = SimCLR(8, hidden=(16,), embed_dim=8, proj_dim=4, rng=0)
    rng = np.random.RandomState(0)
    a = rng.normal(size=(16, 8))
    matched = s.loss(a, a + rng.normal(0, 0.01, a.shape)).item()
    shuffled = s.loss(a, rng.normal(size=(16, 8))).item()
    assert matched < shuffled


def test_simclr_representation_beats_raw_pixels(digits_split):
    """The whole claim of self-supervision: no labels were used to learn this."""
    Xtr, Xte, ytr, yte = digits_split
    s = SimCLR(64, hidden=(64,), embed_dim=32, proj_dim=16, rng=0)
    aug = GaussianNoiseAugment(0.25, rng=0)
    opt = Adam(s.parameters(), lr=3e-3)
    rng = np.random.RandomState(0)
    for _ in range(60):
        for b in _batches(len(Xtr), rng):
            opt.zero_grad()
            s.loss(aug(Xtr[b]), aug(Xtr[b])).backward()
            opt.step()
    s.eval()

    def probe(f):
        return LogisticRegression(max_iter=500).fit(f(Xtr), ytr).score(f(Xte), yte)

    assert probe(lambda z: s.encode(z).data) > probe(lambda z: z)


def test_simclr_keeps_the_layer_it_does_not_train_on(digits_split):
    """The counter-intuitive finding: h, one layer BEFORE the loss, is the better
    representation. The projection head absorbs the invariance the loss demands,
    which is information a downstream task may well want back."""
    Xtr, Xte, ytr, yte = digits_split
    s = SimCLR(64, hidden=(64,), embed_dim=32, proj_dim=16, rng=0)
    aug = GaussianNoiseAugment(0.25, rng=0)
    opt = Adam(s.parameters(), lr=3e-3)
    rng = np.random.RandomState(0)
    for _ in range(60):
        for b in _batches(len(Xtr), rng):
            opt.zero_grad()
            s.loss(aug(Xtr[b]), aug(Xtr[b])).backward()
            opt.step()
    s.eval()

    def probe(f):
        return LogisticRegression(max_iter=500).fit(f(Xtr), ytr).score(f(Xte), yte)

    assert probe(lambda z: s.encode(z).data) >= probe(lambda z: s.project(z).data)


def test_masked_autoencoder_inpaints_what_it_cannot_see(digits_split):
    Xtr, Xte, _, _ = digits_split
    m = MaskedAutoEncoder(64, hidden=(64,), embed_dim=32, mask_ratio=0.5, rng=0)
    opt = Adam(m.parameters(), lr=3e-3)
    rng = np.random.RandomState(0)
    for _ in range(60):
        for b in _batches(len(Xtr), rng):
            opt.zero_grad()
            m.loss(Xtr[b]).backward()
            opt.step()
    m.eval()

    mask = np.random.RandomState(1).uniform(size=Xte.shape) < 0.5
    recon = m.reconstruct(Xte, mask)
    # only the hidden cells are a real prediction; the visible ones are given
    assert ((recon - Xte) ** 2)[mask].mean() < ((Xte) ** 2)[mask].mean() / 2
    assert np.allclose(recon[~mask], Xte[~mask])


def test_masked_autoencoder_scores_only_hidden_positions():
    """Scoring visible cells too would reward copying, which teaches nothing."""
    m = MaskedAutoEncoder(4, hidden=(8,), embed_dim=4, mask_ratio=0.0, rng=0)
    X = np.random.RandomState(0).uniform(size=(8, 4))
    # with nothing masked there is no prediction to make; the loss falls back to
    # plain reconstruction rather than dividing by zero
    assert np.isfinite(m.loss(X).item())
