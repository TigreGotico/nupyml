"""Category encoders and feature engineering: getting raw columns model-ready.

THE ENCODING PROBLEM
--------------------
Most models want numbers, but real data is full of CATEGORIES -- zip codes, job
titles, product ids. One-hot encoding (in ``preprocessing``) works until the
category has hundreds of levels, when it explodes the feature space. The
target-based encoders here map each category to a single number derived from the
target, staying compact -- but that shortcut LEAKS the target if done naively (a
category seen once gets encoded as its own row's label), so each encoder is
really a different way of BORROWING STRENGTH toward a prior to bound that leak.

* ``WOEEncoder`` -- weight of evidence: ``log(odds within category / overall
  odds)``. The standard in credit scoring, because it linearises the relationship
  with the log-odds a logistic model wants.
* ``JamesSteinEncoder`` -- shrink each category's target mean toward the global
  mean by an amount that depends on how RELIABLE that category's estimate is
  (Stein shrinkage). Rare categories shrink hard, frequent ones barely.
* ``MEstimateEncoder`` -- the same shrinkage with a single, simpler smoothing
  parameter -- an additive pseudo-count toward the prior.
* ``LeaveOneOutEncoder`` -- encode each row with its category's mean computed
  WITHOUT that row, which removes the direct leak (the ordered-statistic idea
  from ``ensemble``, applied as an encoder).
* ``BinaryEncoder`` / ``CountEncoder`` -- target-free: represent a category by its
  binary digits (log2 columns instead of one-hot's n) or by its frequency.

FEATURE ENGINEERING
-------------------
* ``Winsorizer`` -- clip extreme values to percentiles, taming outliers without
  dropping rows.
* ``RareLabelEncoder`` -- fold infrequent categories into one "rare" bucket, so
  the model is not distracted by levels it has barely seen.
* ``CyclicalEncoder`` -- encode a cyclical feature (hour, month, angle) as
  ``(sin, cos)`` so that 23:00 and 00:00 are neighbours, which a raw integer
  makes look maximally far apart.

The category_encoders and feature-engine libraries.
"""
from .categorical import (WOEEncoder, JamesSteinEncoder, MEstimateEncoder,
                         LeaveOneOutEncoder, BinaryEncoder, CountEncoder)
from .engineering import Winsorizer, RareLabelEncoder, CyclicalEncoder

__all__ = [
    "WOEEncoder", "JamesSteinEncoder", "MEstimateEncoder", "LeaveOneOutEncoder",
    "BinaryEncoder", "CountEncoder", "Winsorizer", "RareLabelEncoder",
    "CyclicalEncoder",
]
