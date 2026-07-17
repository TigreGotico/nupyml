"""Concept-drift detection: noticing when the data-generating process CHANGES.

THE PROBLEM
-----------
A model trained on last year's data assumes the world stays the same. It does
not: spam evolves, customers shift, sensors age. CONCEPT DRIFT is a change in the
distribution AFTER deployment, and the danger is that it is silent -- accuracy
degrades while the model reports business as usual. A drift detector watches a
stream (of errors, or of a feature) and RAISES A FLAG when the statistics shift
enough that "the process changed" beats "this is just noise".

The hard part is exactly that trade: react to a real change fast, but do not
cry wolf at every unlucky run of noise. Every detector here is a different
statistical answer to when a run of surprising observations is too surprising to
be chance.

THE METHODS
-----------
* ``ADWIN`` -- keep an adaptive window and split it wherever the two halves'
  means differ significantly; the window shrinks the instant drift appears, which
  both detects it and forgets the stale data. The most principled, with a
  guarantee.
* ``DDM`` / ``EDDM`` -- watch a classifier's error rate (DDM) or the spacing
  between errors (EDDM); warn, then alarm, when it worsens by several standard
  deviations from its best.
* ``PageHinkley`` -- a cumulative-sum test: accumulate how far the stream runs
  above its running mean, and alarm when that accumulation exceeds a threshold.
  The classic change-point test.
* ``KSWIN`` -- compare a recent window against an older one with the
  Kolmogorov-Smirnov test; distribution-free, so it catches changes in SHAPE, not
  just the mean.

Gama et al., "A survey on concept drift adaptation" (2014).
"""
from .detectors import ADWIN, DDM, EDDM, PageHinkley, KSWIN

__all__ = ["ADWIN", "DDM", "EDDM", "PageHinkley", "KSWIN"]
