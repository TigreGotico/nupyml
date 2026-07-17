"""Copulas: modelling DEPENDENCE separately from the marginals.

THE IDEA
--------
Sklar's theorem: any joint distribution splits cleanly into two independent
choices -- the MARGINAL distribution of each variable, and a COPULA that couples
them, capturing the dependence structure alone. This separation is powerful
because it lets you model each variable's own distribution however you like
(normal, skewed, heavy-tailed) and then, SEPARATELY, choose how they depend on
each other. Correlation collapses dependence to a single number; a copula
describes its whole shape.

WHY THE SHAPE OF DEPENDENCE MATTERS
-----------------------------------
Two variable pairs can have identical correlation yet completely different
TAIL behaviour -- whether they crash together. That difference is invisible to
correlation and central to risk, which is why copulas run through quantitative
finance (and why misusing the Gaussian copula, which has NO tail dependence, is
often blamed for underpricing the risk behind the 2008 crash).

* ``GaussianCopula`` -- dependence from a correlation matrix, NO tail dependence
  (extremes decouple). Simple, and the cautionary tale.
* ``StudentTCopula`` -- like Gaussian but with SYMMETRIC tail dependence: extreme
  values in both directions tend to co-occur. Heavier tails, controlled by the
  degrees of freedom.
* Archimedean copulas -- ``ClaytonCopula`` (lower-tail dependence: they crash
  together but boom independently), ``GumbelCopula`` (upper-tail), ``FrankCopula``
  (symmetric, no tail). Each captures a different, asymmetric dependence shape
  from a single parameter.

Sklar (1959); Nelsen, *An Introduction to Copulas* (2006).
"""
from .copulas import (GaussianCopula, StudentTCopula, ClaytonCopula,
                     GumbelCopula, FrankCopula)

__all__ = ["GaussianCopula", "StudentTCopula", "ClaytonCopula", "GumbelCopula",
           "FrankCopula"]
