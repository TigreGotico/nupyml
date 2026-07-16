import numpy as np
import pytest

from nupyml import nn
from nupyml.autograd import Tensor, gradcheck
from nupyml.datasets import make_moons, make_regression
from nupyml.model_selection import train_test_split


def test_module_params_and_state_dict(tmp_path):
    model = nn.Sequential(nn.Linear(4, 8, rng=0), nn.ReLU(), nn.Linear(8, 2, rng=1))
    params = list(model.parameters())
    assert len(params) == 4  # 2 weights + 2 biases
    sd = model.state_dict()
    model2 = nn.Sequential(nn.Linear(4, 8, rng=7), nn.ReLU(), nn.Linear(8, 2, rng=8))
    model2.load_state_dict(sd)
    x = np.random.RandomState(0).normal(size=(5, 4))
    assert np.allclose(model(Tensor(x)).data, model2(Tensor(x)).data)
    # save/load roundtrip
    p = tmp_path / "model.npz"
    model.save(p)
    model3 = nn.Sequential(nn.Linear(4, 8, rng=9), nn.ReLU(), nn.Linear(8, 2, rng=10))
    model3.load(p)
    assert np.allclose(model(Tensor(x)).data, model3(Tensor(x)).data)


def test_layer_gradients():
    x = Tensor(np.random.RandomState(0).normal(size=(4, 6)), requires_grad=True)
    for layer in [nn.Linear(6, 3, rng=0), nn.LayerNorm(6), nn.BatchNorm1d(6),
                  nn.Tanh(), nn.Sigmoid(), nn.GELU()]:
        assert gradcheck(lambda a: layer(a), [x], rtol=1e-3, atol=1e-5)


def test_losses_grad():
    logits = Tensor(np.random.RandomState(0).normal(size=(5, 3)), requires_grad=True)
    target = np.array([0, 2, 1, 1, 0])
    assert gradcheck(lambda a: nn.CrossEntropyLoss()(a, target), [logits])
    pred = Tensor(np.random.RandomState(1).uniform(0.1, 0.9, size=(6,)), requires_grad=True)
    tgt = np.array([0, 1, 1, 0, 1, 0], dtype=float)
    assert gradcheck(lambda a: nn.BCELoss()(a, tgt), [pred])
    raw = Tensor(np.random.RandomState(2).normal(size=(6,)), requires_grad=True)
    assert gradcheck(lambda a: nn.BCEWithLogitsLoss()(a, tgt), [raw])
    y = np.random.RandomState(3).normal(size=(5, 1))
    p2 = Tensor(np.random.RandomState(4).normal(size=(5, 1)), requires_grad=True)
    assert gradcheck(lambda a: nn.MSELoss()(a, y), [p2])
    assert gradcheck(lambda a: nn.HuberLoss(delta=0.5)(a, y), [p2])


def test_bce_with_logits_matches_bce():
    rng = np.random.RandomState(0)
    z = rng.normal(size=10)
    y = rng.randint(0, 2, size=10).astype(float)
    from nupyml.utils import sigmoid
    l1 = nn.BCEWithLogitsLoss()(Tensor(z), y).item()
    l2 = nn.BCELoss()(Tensor(sigmoid(z)), y).item()
    assert abs(l1 - l2) < 1e-9


@pytest.mark.parametrize("opt_cls,kwargs", [
    (nn.SGD, {"lr": 0.1}),
    (nn.SGD, {"lr": 0.05, "momentum": 0.9, "nesterov": True}),
    (nn.Adam, {"lr": 0.1}),
    (nn.AdamW, {"lr": 0.1}),
    (nn.RMSprop, {"lr": 0.05}),
])
def test_optimizers_minimize_quadratic(opt_cls, kwargs):
    w = nn.Parameter(np.array([5.0, -3.0]))
    opt = opt_cls([w], **kwargs)
    for _ in range(200):
        opt.zero_grad()
        loss = (w * w).sum()
        loss.backward()
        opt.step()
    assert np.all(np.abs(w.data) < 1e-2)


def test_lr_schedulers():
    w = nn.Parameter(np.zeros(1))
    opt = nn.SGD([w], lr=1.0)
    sched = nn.StepLR(opt, step_size=10, gamma=0.1)
    for _ in range(10):
        sched.step()
    assert abs(opt.lr - 0.1) < 1e-12
    opt2 = nn.SGD([w], lr=1.0)
    cos = nn.CosineAnnealingLR(opt2, T_max=100)
    for _ in range(100):
        cos.step()
    assert opt2.lr < 1e-8


def test_clip_grad_norm():
    w = nn.Parameter(np.ones(4))
    (w * 100.0).sum().backward()
    total = nn.clip_grad_norm([w], max_norm=1.0)
    assert total > 1.0
    assert abs(np.linalg.norm(w.grad) - 1.0) < 1e-9


def test_mlp_classifier_beats_linear_on_moons():
    X, y = make_moons(300, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = nn.MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300,
                           random_state=0)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    assert acc > 0.9, acc
    proba = clf.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1)
    from nupyml.linear_model import LogisticRegression
    lin_acc = LogisticRegression().fit(Xtr, ytr).score(Xte, yte)
    assert acc > lin_acc


def test_mlp_regressor():
    X, y = make_regression(n_samples=300, n_features=5, n_informative=3,
                           noise=0.1, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=0)
    reg = nn.MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=0)
    reg.fit(Xtr, ytr)
    assert reg.score(Xte, yte) > 0.95


def test_batchnorm_train_vs_eval():
    bn = nn.BatchNorm1d(3)
    x = np.random.RandomState(0).normal(5, 2, size=(200, 3))
    bn.train()
    for _ in range(50):
        bn(Tensor(x))
    bn.eval()
    out = bn(Tensor(x)).data
    assert np.all(np.abs(out.mean(axis=0)) < 0.5)


def test_dropout_module():
    d = nn.Dropout(p=0.5, rng=0)
    x = Tensor(np.ones((100, 100)))
    d.train()
    out = d(x)
    assert 0.4 < (out.data == 0).mean() < 0.6
    d.eval()
    assert np.allclose(d(x).data, x.data)
