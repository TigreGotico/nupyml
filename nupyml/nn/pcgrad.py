"""PCGrad: project away CONFLICTING task gradients (Yu et al., 2020)."""
import numpy as np


def pcgrad(task_grads):
    """PCGrad: project away CONFLICTING task gradients (Yu et al., 2020).

    In multi-task learning, gradients from different tasks can point in opposing
    directions -- one task's update undoes another's. PCGrad detects a conflict (a
    negative dot product between two task gradients) and PROJECTS one gradient onto
    the normal plane of the other, removing only the conflicting component. The
    de-conflicted gradients are then summed. This stops tasks from fighting and is
    the standard fix for negative transfer.

    ``task_grads`` is a list of flat gradient vectors (one per task); returns the
    combined gradient.
    """
    import copy
    grads = [np.array(g, float) for g in task_grads]
    n = len(grads)
    proj = [g.copy() for g in grads]
    rng_order = list(range(n))
    for i in range(n):
        for j in rng_order:
            if i == j:
                continue
            dot = proj[i] @ grads[j]
            if dot < 0:                                # conflict -> project out
                proj[i] -= dot / (grads[j] @ grads[j] + 1e-12) * grads[j]
    return np.sum(proj, axis=0)


__all__ = ["pcgrad"]
