"""Metric learning: learn the DISTANCE, instead of assuming Euclidean.

THE PROBLEM WITH EUCLIDEAN DISTANCE
-----------------------------------
Every distance-based method -- kNN, k-means, kernels -- rests on a notion of
"close". The default is Euclidean, which treats all features as equally important
and independent. That is almost never true: one feature may be pure noise,
another may matter ten times as much, two may be redundant. Euclidean distance
weights them all the same and so measures the wrong thing.

THE IDEA
--------
Learn a linear transformation ``L`` of the input and measure distance in the
TRANSFORMED space -- equivalently, a Mahalanobis distance
``(x - y)^T M (x - y)`` with ``M = L^T L``. The transformation is fitted so that
points with the SAME label become close and points with DIFFERENT labels become
far. The learned ``M`` stretches informative directions, shrinks noise ones, and
rotates away redundancy -- exactly the correction Euclidean distance lacks.

THE METHODS
-----------
* ``NCA`` -- Neighbourhood Components Analysis. Learn ``L`` to directly maximise
  a SMOOTH, differentiable proxy for leave-one-out kNN accuracy. Doubles as a
  supervised dimensionality reducer.
* ``LMNN`` -- Large Margin Nearest Neighbours. Pull each point's same-label
  "target neighbours" in, and push different-label "impostors" out beyond a
  margin -- the same large-margin idea as the SVM, applied to a distance.

The learned transform is then handed to an ordinary kNN, which suddenly works far
better because it is finally measuring the right thing.
"""
from .core import NCA, LMNN
from .advanced import ITML, LFDA, RCA

__all__ = ["NCA", "LMNN", "ITML", "LFDA", "RCA"]
