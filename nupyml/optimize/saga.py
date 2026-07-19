"""Variance reduction with a gradient TABLE, no snapshots (Defazio, 2014)."""
import numpy as np


def saga(grad_i, x0, n_samples, lr=0.05, epochs=50, random_state=None):
    """Variance reduction with a gradient TABLE, no snapshots (Defazio, 2014).

    SVRG must periodically recompute a full gradient; SAGA avoids that by keeping a
    TABLE of the most recent gradient seen for each term. Each step picks a term,
    forms the corrected gradient ``g_i(x) - table_i + mean(table)``, updates ``x``,
    and refreshes ``table_i`` -- so the variance reduction is continuous, with no
    epochs of full-gradient computation. Same fast convergence as SVRG, and it also
    supports composite (proximal) objectives naturally. ``grad_i(x, i)`` is term
    ``i``'s gradient.
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    x = np.asarray(x0, float).copy()
    table = np.array([grad_i(x, i) for i in range(n_samples)])
    mean = table.mean(axis=0)
    for _ in range(epochs * n_samples):
        i = rng.randint(n_samples)
        gi = grad_i(x, i)
        x = x - lr * (gi - table[i] + mean)              # variance-reduced step
        mean += (gi - table[i]) / n_samples              # keep the running mean
        table[i] = gi
    return x


__all__ = ["saga"]
