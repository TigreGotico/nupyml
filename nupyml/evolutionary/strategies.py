"""Evolution strategies: adapt the search distribution, not just the points.

THE SHIFT IN THINKING
---------------------
A genetic algorithm maintains a population of POINTS. An evolution strategy
maintains a DISTRIBUTION -- a gaussian -- and the population is merely a sample
from it. Each generation: sample, evaluate, and move the distribution toward
whatever scored well.

That reframing is what makes the family principled rather than a bag of
heuristics. The object being optimized is now ``(mean, covariance)``, and there
is something coherent to say about how to update it.

WHY THE COVARIANCE IS THE WHOLE GAME
------------------------------------
A plain ES samples with a spherical gaussian: equally far in every direction.
On a well-conditioned bowl that is fine. On a narrow curved valley -- which is
what hard problems look like -- it is hopeless. The valley might be a thousand
times longer than it is wide, so a step big enough to make progress along it
overshoots the walls, and a step small enough to stay inside crawls.

This is precisely the ill-conditioning that Newton's method fixes with the
Hessian. CMA-ES does the same thing without derivatives: it LEARNS the shape of
the landscape from which samples scored well, encoding it in a covariance matrix.
The sampling ellipsoid stretches along the valley and flattens across it, and
suddenly the step size that works along the valley is the one being taken.

The covariance approximates the inverse Hessian. CMA-ES is, effectively, a
derivative-free quasi-Newton method -- which is why it works on problems that
defeat every other method here, and why it is the one to reach for on a hard
continuous black box.

THE TWO ADAPTATIONS
-------------------
* **Rank-mu update** -- pull the covariance toward the directions the good
  samples came from THIS generation. Fast, but noisy with a small population.
* **Evolution path (rank-one update)** -- accumulate the mean's movement over
  generations. If it keeps stepping the same way, that direction deserves more
  variance; if the steps zigzag and cancel, they do not. This extracts the
  correlation BETWEEN generations, which a single generation's sample cannot see,
  and it is what lets CMA-ES work with a population far smaller than the
  covariance has entries.

Only the ORDER of the fitnesses is ever used -- never the values. So CMA-ES is
invariant to any monotone transform of the objective: optimizing ``f``, ``f^3``,
and ``log f`` are literally the same run. That is a much stronger guarantee than
most optimizers offer, and it means the method cannot be broken by a badly
scaled objective.

Hansen & Ostermeier (2001).
"""
import numpy as np

from ..utils import check_random_state


class EvolutionStrategy:
    """(mu, lambda)-ES with self-adaptive step size: the simple ancestor.

    Sample ``lambda`` points from an isotropic gaussian, keep the best ``mu``,
    move the mean to their average. The step size adapts by the 1/5th success
    rule: if more than a fifth of samples improve, the steps are too timid --
    grow; otherwise shrink.

    "(mu, lambda)" means the parents are DISCARDED each generation -- the new mean
    comes only from the children. That deliberately allows the mean to move
    somewhere worse, which is what lets it escape a local optimum and climb out of
    a bad basin. "(mu + lambda)" keeps the parents in the running and cannot get
    worse, which converges faster and gets stuck more.

    Read this first, then ``CMAES``, which is this with the isotropy removed.
    """

    def __init__(self, func, x0, sigma=0.5, population_size=None, mu=None,
                 n_generations=200, random_state=None):
        self.func = func
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sigma = sigma
        self.population_size = population_size or 4 + int(3 * np.log(len(self.x0)))
        self.mu = mu or max(1, self.population_size // 2)
        self.n_generations = n_generations
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        mean = self.x0.copy()
        sigma = self.sigma
        self.history_ = []
        best_x, best_f = mean.copy(), self.func(mean)

        for _ in range(self.n_generations):
            samples = mean + sigma * rng.normal(size=(self.population_size,
                                                      len(mean)))
            fitness = np.array([self.func(s) for s in samples])
            order = np.argsort(fitness)

            if fitness[order[0]] < best_f:
                best_f, best_x = float(fitness[order[0]]), samples[order[0]].copy()

            parent_f = self.func(mean)
            # the 1/5th rule: keep the success rate near 0.2 by resizing the step
            success_rate = (fitness < parent_f).mean()
            sigma *= np.exp((success_rate - 0.2) / 0.8 * 0.2)

            mean = samples[order[:self.mu]].mean(axis=0)
            self.history_.append(best_f)

        self.best_, self.best_fitness_, self.sigma_ = best_x, best_f, sigma
        return self


class CMAES:
    """Covariance matrix adaptation evolution strategy.

    The strongest general-purpose derivative-free optimizer for continuous
    problems, and close to parameter-free: the defaults below are the published
    ones and are derived rather than tuned, so ``sigma`` (the initial step, which
    should be roughly a third of the region you believe the optimum lies in) is
    usually the only thing worth setting.

    OPTIMIZATION AND NUMERICAL CARE
    -------------------------------
    Sampling needs a matrix square root of ``C``, which is an eigendecomposition:
    ``O(d^3)``, easily the most expensive step. It is NOT recomputed every
    generation -- ``C`` moves slowly, so refreshing it every ``d/10`` or so
    generations costs nothing in quality and takes the amortised cost to
    ``O(d^2)`` per generation.

    ``C`` is also forced symmetric on every update. It is symmetric in exact
    arithmetic, but floating point drifts, and once it is not symmetric
    ``eigh`` reads only the lower triangle -- silently optimizing a different
    matrix than the one being updated. Symmetrising is one cheap line that
    removes a class of bug that is invisible until it is not.
    """

    def __init__(self, func, x0, sigma=0.5, population_size=None,
                 n_generations=200, random_state=None):
        self.func = func
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sigma = sigma
        self.population_size = population_size
        self.n_generations = n_generations
        self.random_state = random_state

    def _setup(self, n):
        """The published constants. Every one is a function of the dimension --
        none is a tuned magic number, which is why CMA-ES needs no tuning."""
        lam = self.population_size or 4 + int(3 * np.log(n))
        mu = lam // 2

        # log-decreasing recombination weights: the best sample counts most, and
        # the weighting is what makes the mean update a weighted average rather
        # than a plain one
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        # the "variance-effective selection mass": how many samples the weighted
        # mean is effectively averaging. Every learning rate below scales with
        # it, because that is what sets how noisy the update is
        mueff = 1.0 / (w ** 2).sum()

        c = dict(
            lam=lam, mu=mu, w=w, mueff=mueff,
            cc=(4 + mueff / n) / (n + 4 + 2 * mueff / n),   # path for C
            cs=(mueff + 2) / (n + mueff + 5),               # path for sigma
            c1=2 / ((n + 1.3) ** 2 + mueff),                # rank-one rate
            chiN=np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)),
        )
        c["cmu"] = min(1 - c["c1"],
                       2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        c["damps"] = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c["cs"]
        return c

    def run(self):
        rng = check_random_state(self.random_state)
        n = len(self.x0)
        k = self._setup(n)

        mean = self.x0.copy()
        sigma = self.sigma
        C = np.eye(n)
        p_c = np.zeros(n)      # evolution path for C
        p_s = np.zeros(n)      # evolution path for sigma
        B, D = np.eye(n), np.ones(n)
        eigen_age = 0
        # see the class docstring: C changes slowly, so its decomposition is
        # refreshed on a schedule rather than every generation
        eigen_period = max(1, int(n / 10))

        best_x, best_f = mean.copy(), np.inf
        self.history_ = []

        for gen in range(self.n_generations):
            if eigen_age >= eigen_period:
                C = np.triu(C) + np.triu(C, 1).T      # enforce symmetry
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-20))    # a tiny negative eigenvalue
                eigen_age = 0                         # is roundoff, not signal
            eigen_age += 1

            z = rng.normal(size=(k["lam"], n))
            y = z @ (B * D).T                 # correlated: y ~ N(0, C)
            samples = mean + sigma * y
            fitness = np.array([self.func(s) for s in samples])
            order = np.argsort(fitness)       # only the ORDER is ever used

            if fitness[order[0]] < best_f:
                best_f, best_x = float(fitness[order[0]]), samples[order[0]].copy()
            self.history_.append(best_f)

            old_mean = mean
            y_w = k["w"] @ y[order[:k["mu"]]]
            mean = old_mean + sigma * y_w

            # --- step size, via the conjugate evolution path ---
            # p_s accumulates steps in the SPHERICAL frame (C^-1/2 applied), so
            # its length can be compared against a known constant: the expected
            # length under pure randomness. Longer means the steps are aligned
            # and progress is being made, so take bigger ones; shorter means they
            # are cancelling, so take smaller ones.
            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            p_s = (1 - k["cs"]) * p_s + \
                np.sqrt(k["cs"] * (2 - k["cs"]) * k["mueff"]) * (C_inv_sqrt @ y_w)
            sigma *= np.exp((k["cs"] / k["damps"]) *
                            (np.linalg.norm(p_s) / k["chiN"] - 1))

            # --- covariance ---
            # the Heaviside stall: right after a big step sigma is still catching
            # up, and letting p_c accumulate then would inflate C spuriously
            hsig = (np.linalg.norm(p_s) /
                    np.sqrt(1 - (1 - k["cs"]) ** (2 * (gen + 1))) / k["chiN"]
                    ) < (1.4 + 2 / (n + 1))
            p_c = (1 - k["cc"]) * p_c + \
                hsig * np.sqrt(k["cc"] * (2 - k["cc"]) * k["mueff"]) * y_w

            # rank-one: the direction the mean has been consistently moving
            rank_one = np.outer(p_c, p_c)
            # rank-mu: the directions this generation's good samples came from
            y_sel = y[order[:k["mu"]]]
            rank_mu = (y_sel * k["w"][:, None]).T @ y_sel
            # the correction restores the variance the stalled path did not add
            c1a = k["c1"] * (1 - (1 - hsig) * k["cc"] * (2 - k["cc"]))
            C = (1 - c1a - k["cmu"]) * C + k["c1"] * rank_one + k["cmu"] * rank_mu

        self.best_ = best_x
        self.best_fitness_ = best_f
        self.covariance_ = C
        self.sigma_ = sigma
        return self


__all__ = ["CMAES", "EvolutionStrategy"]
