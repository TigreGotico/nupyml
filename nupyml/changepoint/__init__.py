"""Change-point detection: finding WHERE a signal's behaviour changes.

THE PROBLEM
-----------
A time series is rarely stationary: its mean jumps, its variance shifts, its
trend breaks -- at unknown moments. Change-point detection finds those moments.
Unlike the drift detectors in ``drift`` (which watch a live stream and raise a
flag), these work OFFLINE on a whole series and return the exact split points, so
you can segment the signal into homogeneous pieces.

The core tension is the same everywhere: more change-points always fit the data
better (in the limit, one per point fits perfectly), so every method needs a way
to decide when a split is WORTH it -- a penalty per change-point.

THE METHODS
-----------
* ``pelt`` -- exact optimal segmentation under a per-change penalty, made fast
  ``O(n)`` by a pruning rule. The method to reach for when you want the
  provably-best segmentation.
* ``binary_segmentation`` -- greedy: find the single best split, recurse on each
  side. Fast and simple, but greedy (an early wrong split cannot be undone).
* ``cusum`` -- the cumulative-sum test, detecting a shift in mean from when the
  running sum starts drifting. The classic, and the offline sibling of
  Page-Hinkley in ``drift``.
* ``bocpd`` -- Bayesian Online Change-Point Detection: maintain a probability
  distribution over "how long since the last change" and update it each step.
  Gives calibrated UNCERTAINTY about where the changes are, not just point
  estimates.

Truong, Oudre & Vayatis (2020), a review; Killick et al. (2012) for PELT; Adams &
MacKay (2007) for BOCPD.
"""
from .detectors import pelt, binary_segmentation, cusum, bocpd

__all__ = ["pelt", "binary_segmentation", "cusum", "bocpd"]
