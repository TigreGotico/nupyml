"""Optimize by SAMPLING and refitting the elite (Rubinstein, 1997)."""
import numpy as np


def cross_entropy_method(f, x0, sigma0=1.0, pop_size=50, elite_frac=0.2,
                         max_iter=100, tol=1e-8, random_state=None):
    """Optimize by SAMPLING and refitting the elite (Rubinstein, 1997).

    A derivative-free, gradient-free global-ish method: keep a Gaussian over the
    search space, sample a population, keep the best ("elite") fraction, and REFIT
    the Gaussian to just those elites. The distribution marches toward the optimum
    and its variance shrinks as the elites agree. It is robust to noise and
    multimodality, needs no gradient, and is the workhorse behind many
    reinforcement-learning and planning methods (e.g. CEM-based MPC).
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    if rng is None:
        rng = np.random.RandomState()
    mean = np.asarray(x0, float).copy()
    sigma = np.full(len(mean), float(sigma0))
    n_elite = max(1, int(pop_size * elite_frac))
    for _ in range(max_iter):
        samples = rng.normal(mean, sigma, (pop_size, len(mean)))
        scores = np.array([f(s) for s in samples])
        elite = samples[np.argsort(scores)[:n_elite]]    # lowest f = best
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-9                 # refit to the elite
        if np.max(sigma) < tol:
            break
    return mean


__all__ = ["cross_entropy_method"]
