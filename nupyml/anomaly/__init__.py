"""Anomaly detection: scoring how UNUSUAL each point is.

WHAT MAKES IT DIFFERENT FROM CLASSIFICATION
-------------------------------------------
There are (almost) no labels. You cannot train "anomaly vs normal" because
anomalies are rare, varied, and often unseen -- the whole point is to flag ones
you have never encountered. So these methods learn what NORMAL looks like from
mostly-normal data and score deviations from it, unsupervised. Every method here
is a different answer to "unusual compared to what?".

The ``outlier`` module already has the geometric/model-based classics
(IsolationForest, LOF, OneClassSVM, EllipticEnvelope). This module adds the
STATISTICAL and DISTANCE families that dominate the pyod toolkit, which scale
better and often need no tuning:

* ``HBOS`` -- histogram-based: score each feature by how rare its bin is, then
  sum. Assumes feature independence, which makes it extremely fast.
* ``ECOD`` -- empirical-CDF: a point is anomalous if it sits far in the TAIL of
  each feature's distribution. Parameter-free -- nothing to tune.
* ``COPOD`` -- a copula view of the same tail idea, combining per-feature tail
  probabilities.
* ``KNN`` -- distance to the k-th nearest neighbour: far from everything = odd.
* ``CBLOF`` -- cluster first, then score by distance to the nearest large
  cluster: anomalies fall in small or distant clusters.
* ``ABOD`` -- angle-based: for a normal point, neighbours surround it from many
  directions (high angle variance); an outlier sees them all in roughly one
  direction (low variance). Degrades gracefully in high dimensions where
  distances stop discriminating.

The recurring theme: in high dimensions distance-based scores blur together
(the curse again), which is why the CDF- and angle-based methods exist.

Aggarwal, *Outlier Analysis*; the pyod library.
"""
from .detectors import HBOS, ECOD, COPOD, KNN, CBLOF, ABOD
from ._advanced import (LODA, FeatureBaggingDetector, HalfSpaceTrees,
                        MahalanobisDetector, PCAReconstructionDetector,
                        threshold_iqr, threshold_mad, threshold_gesd)
from ._advanced2 import (ExtendedIsolationForest, IsolationKernel, DeepSVDD,
                         energy_score)

__all__ = ["HBOS", "ECOD", "COPOD", "KNN", "CBLOF", "ABOD",
           "LODA", "FeatureBaggingDetector", "HalfSpaceTrees",
           "MahalanobisDetector", "PCAReconstructionDetector",
           "threshold_iqr", "threshold_mad", "threshold_gesd",
           "ExtendedIsolationForest", "IsolationKernel", "DeepSVDD",
           "energy_score"]
