"""G5: residual block, transformer decoder / causal LM, MDN, Siamese network.

Held to their defining behaviour: the residual block preserves shape and passes
gradients; the causal LM overfits and generates a memorised sequence; the MDN
recovers BOTH modes of bimodal data (where a mean-regressor would sit between
them); the Siamese network shares weights across its twin passes.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml import nn


def test_residual_block_preserves_shape_and_has_skip():
    rb = nn.ResidualBlock(8)
    x = Tensor(np.random.RandomState(0).randn(4, 8), requires_grad=True)
    out = rb(x)
    assert out.shape == (4, 8)
    out.sum().backward()
    assert x.grad is not None and np.all(np.isfinite(x.grad))


def test_transformer_decoder_layer_is_causal_shape():
    layer = nn.TransformerDecoderLayer(16, num_heads=4)
    x = Tensor(np.random.RandomState(0).randn(2, 5, 16))
    out = layer(x, mask=nn.causal_mask(5))
    assert out.shape == (2, 5, 16)


def test_causal_lm_overfits_and_generates_sequence():
    vocab = 6
    seqs = np.array([[1, 2, 3, 4, 5]] * 4)
    lm = nn.CausalLanguageModel(vocab, embed_dim=32, num_heads=4, num_layers=2,
                                max_len=16)
    opt = nn.Adam(lm.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    for _ in range(150):
        opt.zero_grad()
        logits = lm(seqs[:, :-1])
        B, T, V = logits.shape
        loss(logits.reshape(B * T, V), seqs[:, 1:].reshape(-1)).backward()
        opt.step()
    # having memorised 1->2->3->4->5, generation from [1,2] should continue it
    gen = lm.generate([1, 2], 3)
    assert gen.tolist() == [1, 2, 3, 4, 5]


def test_mixture_density_network_recovers_both_modes():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (400, 1))
    # bimodal target: +2 or -2 regardless of x -> the MEAN (0) is never correct
    y = np.where(rng.rand(400) < 0.5, 2.0, -2.0) + 0.1 * rng.randn(400)
    mdn = nn.MixtureDensityNetwork(1, n_components=2, hidden=16)
    opt = nn.Adam(mdn.parameters(), lr=0.01)
    for _ in range(400):
        opt.zero_grad(); mdn.nll(X, y).backward(); opt.step()
    _, mu, _ = mdn.forward(X)
    modes = np.sort(mu.data.mean(axis=0))
    assert modes[0] < -1.5 and modes[1] > 1.5         # both modes found, not the mean


def test_mdn_nll_decreases_with_training():
    rng = np.random.RandomState(1)
    X = rng.uniform(-1, 1, (200, 1))
    y = np.sin(3 * X.ravel()) + 0.1 * rng.randn(200)
    mdn = nn.MixtureDensityNetwork(1, n_components=3, hidden=16)
    opt = nn.Adam(mdn.parameters(), lr=0.01)
    first = float(mdn.nll(X, y).data)
    for _ in range(200):
        opt.zero_grad(); mdn.nll(X, y).backward(); opt.step()
    assert float(mdn.nll(X, y).data) < first


def test_siamese_shares_weights_across_twins():
    enc = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 4))
    sm = nn.SiameseNetwork(enc)
    rng = np.random.RandomState(0)
    a = rng.randn(3, 5)
    e1, e2 = sm(a, a.copy())
    # identical inputs through the shared encoder -> identical embeddings
    assert np.allclose(e1.data, e2.data)
    # the Siamese net exposes exactly the encoder's parameters (weights shared)
    assert len(sm.parameters()) == len(list(enc.parameters()))
