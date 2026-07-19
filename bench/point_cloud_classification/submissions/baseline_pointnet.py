"""Baseline: PointNet -- a per-point MLP with a symmetric max-pool."""
import numpy as np

from nupyml.autograd import Tensor
from nupyml.nn import PointNet, CrossEntropyLoss
from nupyml.nn.optim import Adam


def solve(X_train, y_train, X_test):
    net = PointNet(point_dim=3, n_classes=2, hidden=64, rng=0)
    opt = Adam(net.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    rng = np.random.RandomState(0)
    n = len(X_train)
    for _ in range(40):
        idx = rng.permutation(n)[:64]                      # minibatch
        logits = net(X_train[idx])
        loss = loss_fn(logits, y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    logits = net(X_test)
    return np.argmax(logits.data, axis=1)
