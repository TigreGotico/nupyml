"""Meta-learning: models that learn HOW to learn, and features that describe a
learning problem.

Ordinary training fits one task. Meta-learning fits a family of tasks so that a
NEW task can be solved from very little data:

* **Few-shot classifiers** (``PrototypicalNetwork``, ``MatchingNetwork``) learn an
  embedding in which a brand-new class can be recognised from a handful of
  examples -- by comparing to class means, or by attention over the support set.
* **Gradient-based meta-learners** (``Reptile``, ``MAML``) learn an INITIALISATION
  that a few gradient steps can adapt to any task in the family -- so a new task
  needs only a short fine-tune, not training from scratch.
* **Meta-features** (``extract_meta_features``) describe a dataset with numbers
  (size, class balance, feature statistics, a landmark accuracy) so that a
  meta-model can pick an algorithm for it -- the basis of AutoML's warm starts.
"""
from ._fewshot import PrototypicalNetwork, MatchingNetwork
from ._gradient import Reptile, MAML
from ._features import extract_meta_features

__all__ = ["PrototypicalNetwork", "MatchingNetwork", "Reptile", "MAML",
           "extract_meta_features"]
