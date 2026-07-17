"""Dummy estimators: the baselines every real model must beat.

WHY THESE EXIST
---------------
"90% accuracy" means nothing until you know what a model that IGNORES the features
would score. If 90% of the data is one class, predicting that class always ALSO
scores 90% -- and your model has learned nothing. A dummy estimator makes the
right baseline explicit and unavoidable: it predicts using only the target
distribution, never the input, so any real model must clear it to justify itself.

Reaching for a dummy first is a discipline, not a toy. On imbalanced data
especially, the gap between your model and the dummy is the only honest measure of
whether the features carry signal.
"""
import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from .utils import check_array, check_random_state


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Predict without looking at the features -- the classification baseline.

    Strategies, each a different null model:

    * ``most_frequent`` -- always the majority class. The bar accuracy must clear,
      and the one that exposes imbalance: it can score very high while being
      useless.
    * ``stratified`` -- sample predictions from the training class PROPORTIONS.
      The baseline for metrics that reward calibrated randomness.
    * ``uniform`` -- pick a class uniformly at random. The baseline when every
      class should be equally likely a priori.
    * ``prior`` -- predict_proba returns the class priors, predict the argmax.

    If a real classifier cannot beat the relevant one of these, its features add
    nothing.
    """

    def __init__(self, strategy="most_frequent", random_state=None):
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self._most_frequent = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        n = len(check_array(X, dtype=None))
        rng = check_random_state(self.random_state)
        if self.strategy == "most_frequent" or self.strategy == "prior":
            return np.full(n, self._most_frequent)
        if self.strategy == "stratified":
            return rng.choice(self.classes_, size=n, p=self.class_prior_)
        if self.strategy == "uniform":
            return rng.choice(self.classes_, size=n)
        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        n = len(check_array(X, dtype=None))
        # every row gets the class priors -- the honest "no information" forecast
        return np.tile(self.class_prior_, (n, 1))


class DummyRegressor(BaseEstimator, RegressorMixin):
    """Predict a constant -- the regression baseline.

    * ``mean`` -- always the training mean. This is the baseline that makes r^2
      interpretable: r^2 is defined as the improvement over exactly this predictor,
      so a model with r^2 <= 0 is literally worse than DummyRegressor(mean).
    * ``median`` -- the training median, the baseline under absolute error.
    * ``quantile`` -- any quantile, for pinball-loss baselines.
    * ``constant`` -- a value you specify.

    That r^2 is measured against the mean-dummy is why it exists: it is not an
    arbitrary reference, it is THE reference the metric is built on.
    """

    def __init__(self, strategy="mean", constant=None, quantile=0.5):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.strategy == "mean":
            self.constant_ = y.mean()
        elif self.strategy == "median":
            self.constant_ = np.median(y)
        elif self.strategy == "quantile":
            self.constant_ = np.quantile(y, self.quantile)
        elif self.strategy == "constant":
            self.constant_ = self.constant
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")
        return self

    def predict(self, X):
        check_is_fitted(self, "constant_")
        return np.full(len(check_array(X, dtype=None)), self.constant_)


__all__ = ["DummyClassifier", "DummyRegressor"]
