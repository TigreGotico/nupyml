"""Cross-decomposition: finding shared structure between two sets of variables.

THE PROBLEM
-----------
You have TWO blocks of variables measured on the same samples -- spectra and
concentrations, genes and traits, survey answers and outcomes -- and you want to
know how they relate. PCA finds structure within ONE block; regression predicts
one variable from many. These methods find the DIRECTIONS in each block that are
most related to the other.

* ``PLSRegression`` -- find directions in X that best PREDICT Y. The tool of
  choice when X is high-dimensional and collinear (chemometrics, spectroscopy),
  where ordinary regression collapses.
* ``CCA`` -- find directions in X and Y that are maximally CORRELATED, treating
  the two blocks symmetrically. For discovering shared latent structure rather
  than predicting one from the other.

WHY NOT JUST REGRESS
--------------------
When X has more features than samples, or highly collinear ones (adjacent
wavelengths, co-expressed genes), ordinary least squares is unstable or
undefined -- it tries to invert a near-singular matrix. PLS sidesteps this by
projecting X onto a few latent components that are, by construction, both
informative about X AND predictive of Y, then regresses in that small, stable
space. It is regression that builds its own well-conditioned features.
"""
from .core import PLSRegression, CCA, PLSCanonical

__all__ = ["PLSRegression", "CCA", "PLSCanonical"]
