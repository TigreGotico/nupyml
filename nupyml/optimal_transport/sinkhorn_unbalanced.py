"""Unbalanced OT: relax the marginal constraints so mass can be created/destroyed."""
import numpy as np


def sinkhorn_unbalanced(a, b, cost, reg=0.1, reg_marginal=1.0, n_iter=200):
    """Unbalanced OT: relax the marginal constraints so mass can be created/destroyed.

    THE PROBLEM WITH BALANCED OT
    ----------------------------
    Ordinary OT insists every unit of mass in ``a`` is moved onto ``b`` exactly --
    so one outlier point with spurious mass drags the whole plan to explain it.
    Unbalanced OT replaces the hard marginal constraints with a soft KL penalty
    (weight ``reg_marginal``): the plan may leave some mass unmatched if matching
    it would cost more than the penalty. That makes it ROBUST to outliers and
    usable when the two distributions genuinely have different total mass (e.g.
    cell populations of different sizes).

    Solved by a scaling iteration like Sinkhorn, but each marginal update is raised
    to the power ``reg_marginal/(reg_marginal+reg)`` instead of dividing exactly --
    the soft-constraint analogue of the balanced rescaling.

    Chizat et al. (2018). Returns the transport plan.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    K = np.exp(-cost / reg)
    u = np.ones_like(a); v = np.ones_like(b)
    fi = reg_marginal / (reg_marginal + reg)
    for _ in range(n_iter):
        u = (a / (K @ v + 1e-300)) ** fi
        v = (b / (K.T @ u + 1e-300)) ** fi
    return u[:, None] * K * v[None, :]


__all__ = ["sinkhorn_unbalanced"]
