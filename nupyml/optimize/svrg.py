"""SGD with the convergence of full-batch, via a SNAPSHOT (Johnson & Zhang, 2013)."""
import numpy as np


def svrg(grad_i, x0, n_samples, lr=0.05, epochs=50, random_state=None):
    """SGD with the convergence of full-batch, via a SNAPSHOT (Johnson & Zhang, 2013).

    Plain SGD's gradient is noisy, so it must use a decaying step and converges slowly.
    Stochastic Variance-Reduced Gradient fixes this: periodically it computes the FULL
    gradient at a snapshot point, then each cheap step uses ``g_i(x) - g_i(snapshot) +
    full_grad`` -- an unbiased gradient whose variance SHRINKS to zero as ``x``
    approaches the optimum, because the correction cancels the noise. The result is a
    CONSTANT step size and LINEAR convergence on strongly-convex finite sums.
    ``grad_i(x, i)`` is the gradient of term ``i``.
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    x = np.asarray(x0, float).copy()
    for _ in range(epochs):
        snapshot = x.copy()
        full = np.mean([grad_i(snapshot, i) for i in range(n_samples)], axis=0)
        for _ in range(n_samples):
            i = rng.randint(n_samples)
            x = x - lr * (grad_i(x, i) - grad_i(snapshot, i) + full)
    return x


__all__ = ["svrg"]
