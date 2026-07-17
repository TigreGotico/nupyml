"""Recommenders: predicting preferences from a sea of missing values.

WHAT MAKES THE PROBLEM ITS OWN
------------------------------
A recommender's data is a user-by-item matrix that is almost entirely EMPTY --
any one person has rated a vanishing fraction of the catalogue. The task is to
fill the blanks: predict the ratings not seen. Ordinary regression assumes a
feature matrix; here the "features" are the other missing entries, and the signal
is in which cells are filled and how.

THE CENTRAL IDEA: LATENT FACTORS
--------------------------------
Almost everything here rests on one assumption: the giant sparse matrix is
approximately LOW-RANK. A few hidden factors -- genre, tone, era, whatever the
data implies -- explain most preferences. Represent each user and each item by a
short vector in that shared factor space, and a rating is their dot product. The
factors are never named or designed; they are whatever best reconstructs the
observed ratings, and they routinely turn out to be interpretable after the fact.

THE FORK: RATINGS vs CLICKS
---------------------------
* ``MatrixFactorization`` / ``ALS`` optimize for EXPLICIT ratings -- fit the
  numbers people gave. Trained by minimising squared error on observed cells.
* ``BPR`` optimizes for IMPLICIT feedback -- clicks, plays, purchases, where a
  non-interaction is not a "dislike" but a "don't know". It learns to RANK: an
  item you touched should score above one you did not, which is the right target
  when all you have is presence.
* ``FactorizationMachine`` generalises the dot-product model to arbitrary
  features, so context (time, device, demographics) folds in -- the bridge from
  recommendation to general sparse prediction.

Confusing explicit-rating loss with implicit-feedback ranking is the classic
recommender mistake, and the reason both are here side by side.

Koren, Bell & Volinsky (2009); Rendle et al. (2009, 2010).
"""
from .factorization import (MatrixFactorization, ALS, BPR, FactorizationMachine,
                            SVDpp)
from .neighborhood import UserBasedCF, ItemBasedCF, SLIM

__all__ = ["MatrixFactorization", "ALS", "BPR", "FactorizationMachine", "SVDpp",
           "UserBasedCF", "ItemBasedCF", "SLIM"]
