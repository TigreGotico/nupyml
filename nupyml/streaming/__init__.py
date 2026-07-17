"""Online learning: models that update one example at a time, forever.

WHY LEARN ON A STREAM
---------------------
Batch learning assumes a fixed dataset that fits in memory and a distribution
that does not change. Streams break both: the data never stops, it will not fit,
and what it looks like DRIFTS over time (last year's spam is not this year's). A
streaming model must:

* update in ``O(1)`` per example, never revisiting the past;
* use bounded memory regardless of how much data has flowed through;
* adapt when the distribution shifts, rather than clinging to a stale fit.

THE METHODS
-----------
* ``FTRL`` -- online logistic regression that produces SPARSE models, the
  workhorse of large-scale click prediction.
* ``Hedge`` -- combine many experts online with a regret guarantee, no matter how
  adversarial the sequence.
* ``HoeffdingTree`` -- grow a decision tree from a stream, splitting a node only
  once a statistical test says enough data has been seen to be confident.

The thread is the SAME regret-minimisation idea as the bandits in ``rl``: do
nearly as well as the best fixed model in hindsight, while committing to nothing
in advance.
"""
from .online import FTRLProximal, Hedge, OnlineGradientDescent
from .hoeffding import HoeffdingTree

__all__ = ["FTRLProximal", "Hedge", "OnlineGradientDescent", "HoeffdingTree"]
