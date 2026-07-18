"""Explaining a model's predictions: what drove THIS answer, and in general.

WHY THIS IS A SEPARATE CONCERN
------------------------------
A model that predicts well is not the same as a model you understand. A random
forest with 500 trees is accurate and opaque; when it denies a loan, "500 trees
voted" is not an answer anyone can act on or contest. Interpretability methods
extract a human-readable account of a model's behaviour AFTER it is trained,
without changing it.

GLOBAL vs LOCAL -- THE FIRST DISTINCTION
----------------------------------------
* **Global** -- how does the model behave OVERALL? Which features matter across
  all predictions? ``permutation_importance`` and ``partial_dependence`` answer
  this.
* **Local** -- why did the model predict THIS, for THIS input? ``LIME`` and
  ``KernelSHAP`` answer this. A feature can matter globally yet be irrelevant to
  one particular case, and vice versa, so the two questions are genuinely
  different.

THE METHODS
-----------
* ``permutation_importance`` -- shuffle a feature, see how much accuracy drops.
  The honest global importance: it measures what the model ACTUALLY USES, not
  what it was given.
* ``partial_dependence`` / ``ice`` -- how the prediction changes as ONE feature
  sweeps its range, holding others fixed (PDP averages, ICE shows each row).
* ``LIME`` -- fit a simple linear model to the black box's behaviour in a small
  neighbourhood of one point. Local, model-agnostic, intuitive.
* ``KernelSHAP`` -- attribute a prediction to its features via Shapley values
  from cooperative game theory -- the UNIQUE attribution satisfying a set of
  fairness axioms. The rigorous answer, at a computational price.
* ``integrated_gradients`` -- for differentiable models, attribute by integrating
  the gradient along a path from a baseline. What deep-learning saliency should
  have been.

THE WARNING THAT BELONGS ON ALL OF THEM
---------------------------------------
An explanation is of the MODEL, not of the world. If the model learned a
spurious correlation, a faithful explanation will faithfully report that
spurious correlation -- and it will look like insight. These tools tell you what
the model does, never whether what it does is right.
"""
from .attribution import (permutation_importance, partial_dependence, ice,
                          LIME, KernelSHAP, integrated_gradients)
from .glassbox import (ExplainableBoostingRegressor,
                       ExplainableBoostingClassifier)
from .effects import accumulated_local_effects, h_statistic
from .local import surrogate_tree, Anchors, counterfactual
from ._treeshap import TreeSHAP

__all__ = ["permutation_importance", "partial_dependence", "ice", "LIME",
           "KernelSHAP", "integrated_gradients",
           "ExplainableBoostingRegressor", "ExplainableBoostingClassifier",
           "accumulated_local_effects", "h_statistic",
           "surrogate_tree", "Anchors", "counterfactual", "TreeSHAP"]
