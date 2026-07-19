"""A GP on features learned by a NEURAL NET (deep kernel learning, Wilson 2016)."""
from .kernels import Kernel, RBF
from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


class DeepKernelGP(BaseEstimator, RegressorMixin):
    """A GP on features learned by a NEURAL NET (deep kernel learning, Wilson 2016).

    A GP's power is limited by its kernel: RBF measures similarity in the RAW input
    space, so it struggles when "similar for the task" is a nonlinear function of
    the inputs. Deep kernel learning puts a neural feature extractor BEFORE the
    kernel -- ``k(x, x') = k_RBF(g(x), g(x'))`` -- so the GP measures similarity in a
    learned, task-relevant space while keeping the GP's calibrated uncertainty. Here
    the feature net is trained on the regression target, then an exact GP is fit on
    its features (a practical, non-end-to-end deep kernel).
    """

    def __init__(self, hidden=(32, 16), feature_dim=8, alpha=1e-4, epochs=200,
                 lr=0.01, random_state=None):
        self.hidden = hidden
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        from .. import nn
        from ..autograd import Tensor
        from . import GaussianProcessRegressor
        X, y = check_X_y(X, y, y_numeric=True)
        dims = [X.shape[1], *self.hidden, self.feature_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net_ = nn.Sequential(*layers[:-1])       # feature extractor
        head = nn.Linear(self.feature_dim, 1)
        opt = nn.Adam(list(self.net_.parameters()) + list(head.parameters()),
                      lr=self.lr)
        yt = Tensor(y.reshape(-1, 1))
        for _ in range(self.epochs):                  # train the features on y
            opt.zero_grad()
            pred = head(self.net_(Tensor(X)))
            ((pred - yt) ** 2).mean().backward()
            opt.step()
        feats = self.net_(Tensor(X)).data
        self.gp_ = GaussianProcessRegressor(kernel=RBF(), alpha=self.alpha,
                                            optimize=False).fit(feats, y)
        return self

    def _features(self, X):
        from ..autograd import Tensor
        return self.net_(Tensor(check_array(X))).data

    def predict(self, X, return_std=False):
        check_is_fitted(self, "gp_")
        return self.gp_.predict(self._features(X), return_std=return_std)


__all__ = ["DeepKernelGP"]
