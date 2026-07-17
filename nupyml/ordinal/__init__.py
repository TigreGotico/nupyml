"""Ordinal regression -- predicting ordered categories.

* ``OrdinalLogistic`` -- the proportional-odds (cumulative logit) model: one
  weight vector and a set of ordered thresholds, fit by maximum likelihood.
* ``OrdinalRidge`` -- a fast baseline that regresses on the ranks and snaps to
  the nearest class.

Related specialised regressors live nearby: ``linear_model.MultiTaskLasso`` /
``MultiTaskElasticNet`` for joint-support multi-output regression, and
``ensemble.QuantileForest`` for conditional-quantile prediction.
"""
from .regression import OrdinalLogistic, OrdinalRidge

__all__ = ["OrdinalLogistic", "OrdinalRidge"]
