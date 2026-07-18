"""Mathematical optimization solvers -- the machinery under the estimators.

* **Linear / quadratic programming**: ``linprog_simplex``, ``quadratic_program``
  (active-set).
* **Proximal** (non-smooth / L1): ``fista``, ``admm_lasso``, ``soft_threshold``,
  ``lasso_fista``.
* **Iterative linear**: ``conjugate_gradient`` for SPD systems.
* **Derivative-free**: ``nelder_mead``, ``simulated_annealing``,
  ``particle_swarm`` -- when there is no gradient.

Only numpy is used.
"""
from ._solvers import (linprog_simplex, quadratic_program, conjugate_gradient,
                       fista, soft_threshold, lasso_fista, admm_lasso,
                       nelder_mead, simulated_annealing, particle_swarm)
from ._solvers2 import (lbfgs, levenberg_marquardt, frank_wolfe, powell,
                        cross_entropy_method)
from ._solvers3 import (trust_region_newton_cg, interior_point_qp, spsa, owlqn,
                        basin_hopping)
from ._solvers4 import (slsqp, augmented_lagrangian, svrg, saga, mirror_descent,
                        proximal_newton)

__all__ = ["linprog_simplex", "quadratic_program", "conjugate_gradient",
           "fista", "soft_threshold", "lasso_fista", "admm_lasso",
           "nelder_mead", "simulated_annealing", "particle_swarm",
           "lbfgs", "levenberg_marquardt", "frank_wolfe", "powell",
           "cross_entropy_method",
           "trust_region_newton_cg", "interior_point_qp", "spsa", "owlqn",
           "basin_hopping",
           "slsqp", "augmented_lagrangian", "svrg", "saga", "mirror_descent",
           "proximal_newton"]
