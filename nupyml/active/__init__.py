"""Active learning: choosing WHICH examples to label, when labels are expensive.

THE PREMISE
-----------
Unlabelled data is cheap; labels cost money or expert time (a radiologist reading
a scan, a chemist running an assay). If you can only afford to label a few hundred
of a million points, WHICH should you pick? Random sampling wastes the budget on
easy, redundant points. Active learning picks the examples whose labels would
teach the model the most -- and a model trained on a few hundred well-chosen
points can match one trained on thousands of random ones.

The loop: train on what is labelled, use a QUERY STRATEGY to rank the unlabelled
pool, ask an oracle to label the top few, add them, repeat. Everything here is a
different answer to "most informative".

THE STRATEGIES
--------------
* ``uncertainty_sampling`` / ``margin_sampling`` / ``entropy_sampling`` -- query
  where the CURRENT model is least sure (probability near the boundary, small gap
  between the top two classes, high predictive entropy). Cheap and effective; the
  default family.
* ``query_by_committee`` -- train several models and query where they DISAGREE
  most. Disagreement flags genuine ambiguity that any single model's confidence
  might miss.
* ``expected_model_change`` -- query the point that would most change the model
  if labelled. Directly targets learning impact rather than a proxy for it.
* ``core_set`` -- ignore the model entirely: pick points that best COVER the
  input space (greedy k-center), so the labelled set is representative. A
  diversity strategy, the antidote to uncertainty sampling's habit of clustering
  its queries in one ambiguous region.

The tension throughout is uncertainty vs diversity: query only the uncertain and
you re-sample the same confusing corner; query only for coverage and you ignore
where the model actually struggles. Good practice blends them.

Settles, "Active Learning Literature Survey" (2009).
"""
from .strategies import (uncertainty_sampling, margin_sampling,
                        entropy_sampling, query_by_committee,
                        expected_model_change, core_set)

__all__ = ["uncertainty_sampling", "margin_sampling", "entropy_sampling",
           "query_by_committee", "expected_model_change", "core_set"]
