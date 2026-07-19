"""Optimal transport: the cheapest way to turn one distribution into another.

THE IDEA
--------
You have a pile of sand shaped like distribution ``P`` and want to reshape it
into distribution ``Q``. Moving each grain costs (distance moved) x (amount). The
OPTIMAL TRANSPORT plan is the reshaping with the least total cost, and that
minimum cost is the WASSERSTEIN (earth-mover's) distance between ``P`` and ``Q``.

Why this is a better distance than the usual ones: KL divergence and total
variation compare distributions POINTWISE, so two distributions with disjoint
support are "maximally far" no matter how close their masses actually sit.
Wasserstein instead measures how far mass must MOVE, so distributions that are
close in space are close in distance even if they never overlap -- which is
exactly the property that makes it the right loss for generative models (it is
what WGAN in ``nn`` minimises) and for comparing point clouds.

THE COST, AND THE FIX
---------------------
Exact optimal transport is a linear program -- correct but slow (``O(n^3 log n)``).
The breakthrough that made OT practical is ENTROPIC REGULARISATION: add a small
entropy penalty and the problem becomes solvable by ``sinkhorn`` -- alternately
rescaling the rows and columns of a matrix, a handful of cheap matrix-vector
products. It gives a slightly blurred transport plan very fast, and the blur
shrinks with the regularisation.

THE MODULES / FUNCTIONS
-----------------------
* ``sinkhorn`` -- the entropic-OT solver: transport plan and regularised cost.
* ``wasserstein_distance`` -- exact in 1D (a sort), Sinkhorn-based otherwise.
* ``barycenter`` -- the "average" of several distributions under OT -- a
  Wasserstein mean, which looks far more natural than a pointwise average.
* ``ot_domain_adaptation`` -- transport labelled SOURCE points onto an unlabelled
  TARGET distribution, so a classifier trained on the source works on the target.
* ``mmd`` / ``energy_distance`` -- kernel two-sample distances, cheaper cousins
  used to test whether two samples come from the same distribution.

Peyre & Cuturi, *Computational Optimal Transport* (2019); Cuturi (2013).
"""
from .core import (sinkhorn, wasserstein_distance, barycenter,
                   ot_domain_adaptation, mmd, energy_distance)
from .sinkhorn_unbalanced import sinkhorn_unbalanced
from .sliced_wasserstein import sliced_wasserstein
from .gromov_wasserstein import gromov_wasserstein
from .ot_mapping import ot_mapping

__all__ = ["sinkhorn", "wasserstein_distance", "barycenter",
           "ot_domain_adaptation", "mmd", "energy_distance",
           "sinkhorn_unbalanced", "sliced_wasserstein", "gromov_wasserstein",
           "ot_mapping"]
