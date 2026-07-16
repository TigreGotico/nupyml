"""sklearn-style MLP estimators built on the autograd engine."""
import numpy as np

from ..autograd import Tensor, no_grad
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, check_random_state
from .module import Sequential
from .layers import Linear, ReLU, Tanh
from .losses import CrossEntropyLoss, MSELoss
from .optim import Adam, SGD
from .data import DataLoader

_ACTIVATIONS = {"relu": ReLU, "tanh": Tanh}


class _BaseMLP(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver="adam", alpha=1e-4, batch_size=32, learning_rate=1e-3,
                 max_iter=200, tol=1e-4, n_iter_no_change=10,
                 early_stopping=False, validation_fraction=0.1,
                 warm_start=False, random_state=None, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, n_in, n_out, rng):
        act = _ACTIVATIONS[self.activation]
        layers = []
        prev = n_in
        for h in self.hidden_layer_sizes:
            layers += [Linear(prev, h, rng=rng), act()]
            prev = h
        layers.append(Linear(prev, n_out, rng=rng))
        return Sequential(*layers)

    def _make_optimizer(self, params):
        if self.solver == "adam":
            return Adam(params, lr=self.learning_rate, weight_decay=self.alpha)
        if self.solver == "sgd":
            return SGD(params, lr=self.learning_rate, momentum=0.9,
                       weight_decay=self.alpha)
        if self.solver == "lbfgs":
            return None  # handled separately in _fit_loop
        raise ValueError(f"Unknown solver: {self.solver!r}")

    def _fit_lbfgs(self, X, y, model, loss_fn):
        """Full-batch training through scipy's L-BFGS-B on flattened params."""
        import scipy.optimize
        params = list(model.parameters())
        shapes = [p.data.shape for p in params]
        sizes = [p.data.size for p in params]

        def set_flat(theta):
            i = 0
            for p, shape, size in zip(params, shapes, sizes):
                p.data = theta[i:i + size].reshape(shape).copy()
                i += size

        def loss_grad(theta):
            set_flat(theta)
            model.zero_grad()
            loss = loss_fn(model(Tensor(X)), y)
            reg = 0.5 * self.alpha * sum(float((p.data ** 2).sum())
                                         for p in params)
            loss.backward()
            grad = np.concatenate([
                (p.grad + self.alpha * p.data).ravel() if p.grad is not None
                else (self.alpha * p.data).ravel() for p in params])
            return loss.item() + reg, grad

        theta0 = np.concatenate([p.data.ravel() for p in params])
        res = scipy.optimize.minimize(loss_grad, theta0, jac=True,
                                      method="L-BFGS-B",
                                      options={"maxiter": self.max_iter})
        set_flat(res.x)
        self.loss_curve_ = [float(res.fun)]
        self.n_iter_ = int(res.nit)

    def _fit_loop(self, X, y, model, loss_fn):
        if self.solver == "lbfgs":
            self._fit_lbfgs(X, y, model, loss_fn)
            return
        rng = check_random_state(self.random_state)
        opt = self._make_optimizer(model.parameters())
        if self.early_stopping:
            n_val = max(1, int(self.validation_fraction * len(X)))
            perm = rng.permutation(len(X))
            val, tr = perm[:n_val], perm[n_val:]
            X_val, y_val = X[val], np.asarray(y)[val]
            X, y = X[tr], np.asarray(y)[tr]
        else:
            X_val = None
        loader = DataLoader(X, y, batch_size=self.batch_size, random_state=rng)
        best = np.inf
        stall = 0
        if not (self.warm_start and hasattr(self, "loss_curve_")):
            self.loss_curve_ = []
        self.validation_scores_ = []
        for epoch in range(self.max_iter):
            total, count = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(Tensor(xb)), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                count += len(xb)
            epoch_loss = total / count
            self.loss_curve_.append(epoch_loss)
            if self.verbose:
                print(f"epoch {epoch}: loss={epoch_loss:.6f}")
            if X_val is not None:
                with no_grad():
                    val_loss = loss_fn(model(Tensor(X_val)), y_val).item()
                self.validation_scores_.append(val_loss)
                monitored = val_loss
            else:
                monitored = epoch_loss
            if monitored < best - self.tol:
                best = monitored
                stall = 0
            else:
                stall += 1
                if stall >= self.n_iter_no_change:
                    break
        self.n_iter_ = len(self.loss_curve_)


class MLPClassifier(_BaseMLP, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)
        if not (self.warm_start and hasattr(self, "model_")):
            self.model_ = self._build(X.shape[1], len(self.classes_), rng)
        self.model_.train()
        self._fit_loop(X, y_idx, self.model_, CrossEntropyLoss())
        return self

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X)
        self.model_.eval()
        with no_grad():
            return self.model_(Tensor(X)).data

    def predict_proba(self, X):
        from ..utils import softmax
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class MLPRegressor(_BaseMLP, RegressorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        self._y_mean = y.mean()
        self._y_std = y.std() or 1.0
        y_scaled = ((y - self._y_mean) / self._y_std)[:, None]
        if not (self.warm_start and hasattr(self, "model_")):
            self.model_ = self._build(X.shape[1], 1, rng)
        self.model_.train()
        self._fit_loop(X, y_scaled, self.model_, MSELoss())
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X)
        self.model_.eval()
        with no_grad():
            out = self.model_(Tensor(X)).data.ravel()
        return out * self._y_std + self._y_mean


__all__ = ["MLPClassifier", "MLPRegressor"]
