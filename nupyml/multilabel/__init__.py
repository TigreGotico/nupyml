"""Multi-label classification: each example can carry SEVERAL labels at once.

THE PROBLEM
-----------
Ordinary classification assigns one label per example. But a photo can be tagged
"beach" AND "sunset" AND "friends"; an article can be "politics" AND "economics".
The target is now a SET of labels (a binary vector), and the labels are usually
CORRELATED -- "beach" and "sunset" co-occur, "beach" and "snow" do not. The whole
game is whether, and how, a method exploits those correlations.

THE METHODS, ORDERED BY HOW MUCH CORRELATION THEY CAPTURE
---------------------------------------------------------
* ``BinaryRelevance`` -- train one independent binary classifier per label. Dead
  simple, embarrassingly parallel, and it IGNORES label correlations entirely
  (each label decided alone). The baseline everything else improves on.
* ``ClassifierChain`` -- also one classifier per label, but chained: each sees
  the PREDICTIONS of the earlier labels as extra features, so it can learn "given
  this is a beach photo, is it also a sunset?". Captures correlations at almost
  no extra cost; the order of the chain matters.
* ``LabelPowerset`` -- treat each OBSERVED combination of labels as a single
  class and do ordinary multiclass. Captures correlations perfectly by
  construction, but the number of classes explodes and unseen combinations can
  never be predicted.
* ``RAkEL`` -- an ensemble of label-powersets over small RANDOM subsets of
  labels, voting. It gets powerset's correlation-awareness without its blow-up,
  by keeping each powerset small.
* ``MLkNN`` -- k-nearest-neighbours with a Bayesian twist: predict each label
  from how many of a point's neighbours carry it. Naturally multi-label, and a
  strong lazy baseline.

Read them in order: it is a ladder from "ignore correlations" to "model them
fully but expensively", and choosing a rung is the real decision.

Zhang & Zhou (2014), a review; Read et al. (2011) for chains; Tsoumakas et al.
(2011) for RAkEL.
"""
from .classifiers import (BinaryRelevance, ClassifierChain, LabelPowerset,
                          RAkEL, MLkNN)

__all__ = ["BinaryRelevance", "ClassifierChain", "LabelPowerset", "RAkEL",
           "MLkNN"]
