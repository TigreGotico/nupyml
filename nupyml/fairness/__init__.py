"""Fairness: measuring and mitigating group bias in classifiers.

Two halves:

* ``metrics`` -- group-fairness measures (demographic parity, disparate impact,
  equal opportunity, equalized odds, predictive parity). They answer DIFFERENT
  questions and provably conflict, so choosing one is a policy decision.
* ``mitigation`` -- interventions at each pipeline stage: ``Reweighing`` and
  ``CorrelationRemover`` (pre-processing), ``ExponentiatedGradientReduction``
  (in-processing), ``ThresholdOptimizer`` (post-processing).

Fairness metrics and mitigators take a ``sensitive`` attribute alongside the
usual arguments. Only numpy/scipy are used.
"""
from .metrics import (selection_rate, demographic_parity_difference,
                      disparate_impact_ratio, equal_opportunity_difference,
                      equalized_odds_difference, predictive_parity_difference)
from .mitigation import (Reweighing, CorrelationRemover, ThresholdOptimizer,
                        ExponentiatedGradientReduction)

__all__ = ["selection_rate", "demographic_parity_difference",
           "disparate_impact_ratio", "equal_opportunity_difference",
           "equalized_odds_difference", "predictive_parity_difference",
           "Reweighing", "CorrelationRemover", "ThresholdOptimizer",
           "ExponentiatedGradientReduction"]
