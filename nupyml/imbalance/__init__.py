"""Class imbalance: when one class is rare and accuracy lies.

THE PROBLEM
-----------
Fraud is 0.1% of transactions; a model that predicts "not fraud" always scores
99.9% accuracy and catches nothing. Accuracy is meaningless under imbalance,
and, worse, most learners OPTIMISE it -- so they learn to ignore the rare class
that is usually the one you care about. Two fixes, at two different places:

* **Resample the data** (this module) so the classes are balanced before
  training. Model-agnostic: fix the data, use any learner unchanged.
* **Reweight the loss** (elsewhere: ``class_weight``, focal loss) so the rare
  class costs more to misclassify. Fix the objective, leave the data alone.

THE RESAMPLING FAMILY
---------------------
* ``RandomOverSampler`` / ``RandomUnderSampler`` -- duplicate the minority or drop
  the majority. Trivial, and each has a clear failure: duplication invites
  overfitting to the exact minority points, dropping discards real information.
* ``SMOTE`` -- SYNTHESISE new minority points by interpolating between real ones,
  instead of copying. The idea that made resampling respectable.
* ``ADASYN`` -- SMOTE that makes more synthetic points where the minority is HARD
  (near the boundary), concentrating effort where the classifier struggles.
* ``TomekLinks`` / ``NearMiss`` -- CLEAN the majority: remove points that sit on
  top of the boundary and blur it, rather than removing at random.

THE ONE RULE THAT IS ALWAYS BROKEN
----------------------------------
Resample the TRAINING data only, AFTER splitting off the test set. Oversampling
before the split leaks synthetic copies of test-adjacent points into training and
produces gloriously optimistic, entirely fake scores. This is the single most
common imbalance mistake, and no amount of clever synthesis rescues you from it.
"""
from .resample import (RandomOverSampler, RandomUnderSampler, SMOTE, ADASYN,
                       TomekLinks, NearMiss, BorderlineSMOTE)
from .ensembles import BalancedRandomForest, RUSBoost, EasyEnsemble

__all__ = ["RandomOverSampler", "RandomUnderSampler", "SMOTE", "ADASYN",
           "TomekLinks", "NearMiss", "BorderlineSMOTE",
           "BalancedRandomForest", "RUSBoost", "EasyEnsemble"]
