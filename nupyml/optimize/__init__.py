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
from .linprog_simplex import linprog_simplex
from .quadratic_program import quadratic_program
from .conjugate_gradient import conjugate_gradient
from .fista import fista
from .soft_threshold import soft_threshold
from .lasso_fista import lasso_fista
from .admm_lasso import admm_lasso
from .nelder_mead import nelder_mead
from .simulated_annealing import simulated_annealing
from .particle_swarm import particle_swarm
from .lbfgs import lbfgs
from .levenberg_marquardt import levenberg_marquardt
from .frank_wolfe import frank_wolfe
from .powell import powell
from .cross_entropy_method import cross_entropy_method
from .trust_region_newton_cg import trust_region_newton_cg
from .interior_point_qp import interior_point_qp
from .spsa import spsa
from .owlqn import owlqn
from .basin_hopping import basin_hopping
from .slsqp import slsqp
from .augmented_lagrangian import augmented_lagrangian
from .svrg import svrg
from .saga import saga
from .mirror_descent import mirror_descent
from .proximal_newton import proximal_newton
from .nonlinear_conjugate_gradient import nonlinear_conjugate_gradient
from .subgradient_method import subgradient_method
from .pattern_search import pattern_search
from .douglas_rachford import douglas_rachford
from .chambolle_pock import chambolle_pock

__all__ = ["linprog_simplex", "quadratic_program", "conjugate_gradient",
           "fista", "soft_threshold", "lasso_fista", "admm_lasso",
           "nelder_mead", "simulated_annealing", "particle_swarm",
           "lbfgs", "levenberg_marquardt", "frank_wolfe", "powell",
           "cross_entropy_method",
           "trust_region_newton_cg", "interior_point_qp", "spsa", "owlqn",
           "basin_hopping",
           "slsqp", "augmented_lagrangian", "svrg", "saga", "mirror_descent",
           "proximal_newton",
           "nonlinear_conjugate_gradient", "subgradient_method", "pattern_search",
           "douglas_rachford", "chambolle_pock"]
