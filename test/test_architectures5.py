"""K6: neural architectures v5 -- ViT, NTM, Glow, DDIM, relative attention.

The Vision Transformer classifies digit patches; the Neural Turing Machine learns
the copy task via its external memory; Glow is exactly invertible and fits a
density; DDIM samples a trained diffusion model deterministically and recovers its
distribution; relative-position attention produces a Toeplitz logit matrix
(distance-only bias) for identical tokens.
"""
import numpy as np
import pytest

from sklearn.datasets import load_digits

from nupyml.autograd import Tensor
from nupyml.autograd import functional as F
from nupyml.nn import Adam, DDPM
from nupyml.nn import (VisionTransformer, NeuralTuringMachine, Glow, ddim_sample,
                       RelativePositionAttention)
from nupyml.model_selection import train_test_split
from nupyml.metrics import accuracy_score


def test_vision_transformer_classifies_digits():
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 8, 8)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    vit = VisionTransformer(img_size=8, patch_size=4, n_classes=10, embed_dim=32,
                            depth=2, n_heads=4, rng=0)
    opt = Adam(vit.parameters(), lr=0.005)
    rng = np.random.RandomState(0)
    for _ in range(400):
        idx = rng.randint(0, len(Xtr), 64)
        logits = vit(Xtr[idx])
        onehot = np.eye(10)[ytr[idx]]
        loss = -(F.log_softmax(logits, axis=1) * Tensor(onehot)).sum(axis=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert accuracy_score(yte, vit(Xte).data.argmax(1)) > 0.85


def test_ntm_learns_copy_task():
    ntm = NeuralTuringMachine(n_slots=16, slot_dim=8, input_dim=4,
                              controller_dim=32, rng=0)
    opt = Adam(ntm.parameters(), lr=0.01)
    rng = np.random.RandomState(0)
    for _ in range(800):
        seq = rng.randint(0, 2, (16, 3, 4)).astype(float)
        inp = ([Tensor(seq[:, i, :]) for i in range(3)]
               + [Tensor(np.zeros((16, 4))) for _ in range(3)])
        outs = ntm(inp)
        loss = sum(((outs[3 + i] - Tensor(seq[:, i, :])) ** 2).mean()
                   for i in range(3))
        opt.zero_grad(); loss.backward(); opt.step()
    seq = rng.randint(0, 2, (16, 3, 4)).astype(float)
    inp = ([Tensor(seq[:, i, :]) for i in range(3)]
           + [Tensor(np.zeros((16, 4))) for _ in range(3)])
    outs = ntm(inp)
    recon = np.stack([(outs[3 + i].data > 0.5).astype(float) for i in range(3)], 1)
    assert (recon == seq).mean() > 0.75            # recalls the stored sequence


def test_glow_is_invertible_and_fits_density():
    rng = np.random.RandomState(0)
    g = Glow(dim=4, n_flows=3, hidden=16, rng=0)
    x = rng.randn(8, 4)
    z, _ = g.forward(Tensor(x))
    assert np.allclose(g.inverse(z.data), x, atol=1e-6)      # exact round trip
    # a trained Glow assigns higher density to in-distribution than far points
    data = rng.randn(400, 4) * 0.5 + 3.0
    opt = Adam(g.parameters(), lr=0.01)
    for _ in range(200):
        idx = rng.randint(0, len(data), 64)
        opt.zero_grad(); g.nll(Tensor(data[idx])).backward(); opt.step()
    ll_in = g.log_prob(Tensor(data[:50])).data.mean()
    ll_out = g.log_prob(Tensor(rng.randn(50, 4) * 0.5 - 3.0)).data.mean()
    assert ll_in > ll_out


def test_ddim_is_deterministic_and_recovers_distribution():
    rng = np.random.RandomState(0)
    data = rng.randn(600, 2) * 0.3 + np.array([2.0, -1.0])
    ddpm = DDPM(n_features=2, T=50, hidden=(32, 32), rng=0)
    opt = Adam(ddpm.parameters(), lr=0.005)
    for _ in range(500):
        idx = rng.randint(0, len(data), 64)
        opt.zero_grad(); ddpm.loss(data[idx]).backward(); opt.step()
    s1 = ddim_sample(ddpm, 400, n_steps=20, eta=0.0, rng=np.random.RandomState(5))
    s2 = ddim_sample(ddpm, 400, n_steps=20, eta=0.0, rng=np.random.RandomState(5))
    assert np.allclose(s1, s2)                     # eta=0 -> deterministic
    assert np.allclose(s1.mean(0), [2.0, -1.0], atol=0.4)   # recovers the data


def test_relative_attention_logits_are_toeplitz():
    rpa = RelativePositionAttention(dim=8, max_len=20, rng=0)
    n = 10
    # identical tokens -> content scores are all equal, so the logit matrix is
    # exactly the (distance-only) relative bias -> constant along diagonals
    x = np.tile(np.random.RandomState(1).randn(8), (1, n, 1))
    Q = rpa.q(Tensor(x)); K = rpa.k(Tensor(x))
    scores = (Q @ K.transpose(0, 2, 1)).data[0] / np.sqrt(8)
    bias = rpa.rel_bias.data[rpa._bias_matrix(n).ravel()].reshape(n, n)
    logits = scores + bias
    for i in range(n - 1):
        for j in range(n - 1):
            assert abs(logits[i, j] - logits[i + 1, j + 1]) < 1e-9
    # and it is a working attention layer
    out = rpa(Tensor(np.random.randn(2, n, 8)))
    assert out.shape == (2, n, 8)
    out.sum().backward()
