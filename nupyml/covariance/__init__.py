"""Covariance estimation: the sample covariance is worse than it looks.

WHY THIS IS A WHOLE MODULE
--------------------------
"Just compute the covariance matrix" seems trivial -- until you have more
features than samples, or nearly so. Then the sample covariance is a disaster:

* It is SINGULAR (non-invertible) when features outnumber samples, so anything
  that needs its inverse -- Mahalanobis distance, LDA, a Gaussian likelihood,
  portfolio weights -- simply breaks.
* Even when invertible, its extreme eigenvalues are wildly OVER-estimated and its
  small ones UNDER-estimated. The inverse amplifies exactly the smallest, noisiest
  eigenvalues, so the estimate is not just imprecise but confidently wrong in the
  direction that matters most.

The methods here are the fixes, and they split into two ideas:

SHRINKAGE -- pull the estimate toward a simple target
-----------------------------------------------------
* ``LedoitWolf`` -- blend the sample covariance with a scaled identity, at the
  shrinkage intensity that PROVABLY minimises expected error. No cross-validation:
  the optimal amount is computed in closed form from the data.
* ``OAS`` -- the same idea with a shrinkage formula tuned for Gaussian data,
  often better when that assumption roughly holds.

SPARSITY / ROBUSTNESS -- change what you estimate
-------------------------------------------------
* ``GraphicalLasso`` -- estimate a SPARSE inverse covariance (precision) matrix.
  A zero in the precision matrix means two variables are conditionally
  INDEPENDENT, so this learns the graph of direct relationships, not just
  correlations.
* ``MinCovDet`` -- a robust estimate that ignores outliers by fitting the
  covariance to the most tightly-packed subset of the data.

The through-line: the naive estimator is unbiased but high-variance, and trading
a little bias for a lot less variance -- Stein's insight -- wins decisively here.
"""
from .core import (EmpiricalCovariance, LedoitWolf, OAS, GraphicalLasso,
                   MinCovDet, ShrunkCovariance)

__all__ = ["EmpiricalCovariance", "LedoitWolf", "OAS", "GraphicalLasso",
           "MinCovDet", "ShrunkCovariance"]
