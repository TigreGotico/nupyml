"""M6: neural architectures v7 -- CNP, PointNet, energy-based model, SimSiam, VGAE.

The CNP's prediction improves with more context; PointNet classifies point clouds and
is permutation-invariant; the energy model assigns lower energy in-distribution;
SimSiam learns representations without collapsing; VGAE embeds a graph so that
same-community nodes are closer.
"""
import numpy as np
import pytest

from nupyml.autograd import Tensor
from nupyml.autograd import functional as F
from nupyml.nn import (ConditionalNeuralProcess, PointNet, EnergyBasedModel,
                       SimSiam, VGAE, Adam)


def test_cnp_prediction_improves_with_context():
    rng = np.random.RandomState(0)
    cnp = ConditionalNeuralProcess(x_dim=1, y_dim=1, hidden=64, r_dim=64, rng=0)
    opt = Adam(cnp.parameters(), lr=0.003)
    for _ in range(1500):
        amp = rng.uniform(0.5, 2); ph = rng.uniform(0, 3.14)
        xs = np.sort(rng.uniform(-3, 3, 20)).reshape(-1, 1); ys = amp * np.sin(xs + ph)
        nc = rng.randint(3, 15)
        opt.zero_grad()
        cnp.loss(Tensor(xs[:nc]), Tensor(ys[:nc]), Tensor(xs), Tensor(ys)).backward()
        opt.step()
    mse2, mse10 = [], []
    for _ in range(50):
        amp = rng.uniform(0.5, 2); ph = rng.uniform(0, 3.14)
        xs = np.sort(rng.uniform(-3, 3, 15)).reshape(-1, 1); ys = amp * np.sin(xs + ph)
        m2, _ = cnp.forward(Tensor(xs[:2]), Tensor(ys[:2]), Tensor(xs))
        m10, _ = cnp.forward(Tensor(xs[:10]), Tensor(ys[:10]), Tensor(xs))
        mse2.append(np.mean((m2.data - ys) ** 2))
        mse10.append(np.mean((m10.data - ys) ** 2))
    assert np.mean(mse10) < np.mean(mse2)


def _cloud(shape, rng, n=64):
    if shape == 'sphere':
        p = rng.randn(n, 3); return p / np.linalg.norm(p, axis=1, keepdims=True)
    return rng.uniform(-1, 1, (n, 3))


def test_pointnet_classifies_and_is_permutation_invariant():
    rng = np.random.RandomState(0)
    pn = PointNet(point_dim=3, n_classes=2, hidden=64, rng=0)
    opt = Adam(pn.parameters(), lr=0.005)
    for _ in range(300):
        batch = np.stack([_cloud('sphere' if i % 2 == 0 else 'cube', rng)
                          for i in range(16)])
        y = np.array([0, 1] * 8)
        loss = -(F.log_softmax(pn(Tensor(batch)), axis=1)
                 * Tensor(np.eye(2)[y])).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    test = np.stack([_cloud('sphere' if i % 2 == 0 else 'cube', rng)
                     for i in range(20)])
    ty = np.array([0, 1] * 10)
    assert (pn(Tensor(test)).data.argmax(1) == ty).mean() > 0.9
    # shuffling the points does not change the output (symmetric pooling)
    c = _cloud('sphere', rng)
    o1 = pn(Tensor(c[None])).data
    o2 = pn(Tensor(c[rng.permutation(64)][None])).data
    assert np.allclose(o1, o2, atol=1e-5)


def test_energy_model_lower_energy_in_distribution():
    from nupyml.nn.optim import clip_grad_norm
    rng = np.random.RandomState(0)
    data = rng.randn(500, 2) * 0.4 + np.array([2., -1])
    ebm = EnergyBasedModel(dim=2, hidden=16, langevin_steps=20, langevin_lr=0.02,
                           rng=0)
    opt = Adam(ebm.parameters(), lr=0.0005)
    for _ in range(500):
        idx = rng.randint(0, len(data), 64)
        opt.zero_grad(); ebm.loss(Tensor(data[idx]), reg=2.0, rng=rng).backward()
        clip_grad_norm(ebm.parameters(), 1.0)            # EBM training needs clipping
        opt.step()
    e_in = float(ebm.energy(Tensor(data[:60])).data.mean())
    # average energy over several off-distribution regions is higher than in-dist
    e_out = np.mean([float(ebm.energy(Tensor(rng.randn(60, 2) * 0.4 + c)).data.mean())
                     for c in ([-2, 2], [0, 3], [-3, -3], [4, 4])])
    assert e_in < e_out


def test_simsiam_does_not_collapse():
    rng = np.random.RandomState(0)
    centers = rng.randn(4, 8) * 3
    views = lambda c, n: c + rng.randn(n, 8) * 0.3
    ss = SimSiam(in_dim=8, hidden=64, proj_dim=32, rng=0)
    opt = Adam(ss.parameters(), lr=0.01)
    for _ in range(300):
        c = centers[rng.randint(4)]
        opt.zero_grad()
        ss.loss(Tensor(views(c, 16)), Tensor(views(c, 16))).backward()
        opt.step()
    emb = np.array([ss.encode(Tensor(views(centers[i % 4], 1))).data[0]
                    for i in range(40)])
    assert emb.std() > 0.5                               # representations did not collapse


def test_vgae_recovers_communities():
    rng = np.random.RandomState(0)
    n = 30
    lab = np.array([0] * 15 + [1] * 15)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.4 if lab[i] == lab[j] else 0.05
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    X = np.eye(n)
    vg = VGAE(in_dim=n, hidden=32, latent=8, rng=0)
    opt = Adam(vg.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad(); vg.loss(X, A).backward(); opt.step()
    z = vg.embed(X, A)
    cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    same = np.mean([cos(z[i], z[j]) for i in range(15) for j in range(i + 1, 15)])
    diff = np.mean([cos(z[i], z[j]) for i in range(15) for j in range(15, 30)])
    assert same > diff
