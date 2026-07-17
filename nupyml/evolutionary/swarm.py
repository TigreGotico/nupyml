"""Differential evolution and particle swarms: population methods that work.

Both are simpler than a genetic algorithm, have fewer knobs, and beat it on
continuous problems. Included together because they answer the same question --
"which direction should I try?" -- by reading it off the population rather than
inventing it.
"""
import numpy as np

from ..utils import check_random_state


class DifferentialEvolution:
    """DE: use the population's own spread as the mutation step.

    THE IDEA
    --------
    The mutation is the whole method, and it is one line::

        trial = a + F * (b - c)      for three random distinct individuals

    The step is a DIFFERENCE BETWEEN EXISTING MEMBERS. That is what makes it
    clever: no step size is ever specified, because the population supplies it.

    Early on, when members are scattered, the differences are large and the steps
    explore. As the population converges the differences shrink and the same
    formula fine-tunes. The step size adapts for free, and -- more subtly -- so
    does its SHAPE: if the population has settled into a narrow diagonal valley,
    the difference vectors lie along that valley, so the trial steps do too.
    The method discovers the landscape's correlation structure without ever
    representing it, which is what CMA-ES spends a covariance matrix to do.

    SELECTION IS PAIRWISE
    ---------------------
    Each trial competes ONLY against the parent that produced it, replacing it if
    better. Not against the population, not against the best. So a good solution
    is never displaced by a better one elsewhere, and diversity survives -- DE
    does not suffer the takeover that fitness-proportional selection invites.
    It also makes the best-so-far monotone without needing explicit elitism.

    THE TWO PARAMETERS
    ------------------
    ``F`` (0.5-0.9) scales the difference. ``CR`` is the probability of taking
    each coordinate from the trial. Low ``CR`` changes few coordinates at a time,
    which suits separable problems; high ``CR`` changes most of them, which suits
    problems where the variables interact. That is the one real choice here.

    Storn & Price (1997).
    """

    def __init__(self, func, bounds, population_size=None, F=0.8, CR=0.9,
                 n_generations=200, strategy="rand1bin", random_state=None):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.n_generations = n_generations
        self.strategy = strategy
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        dim = len(self.bounds)
        n = self.population_size or max(15, 10 * dim)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]

        pop = rng.uniform(lo, hi, size=(n, dim))
        fitness = np.array([self.func(p) for p in pop])
        self.history_ = [fitness.min()]

        for _ in range(self.n_generations):
            for i in range(n):
                # three distinct others, none of them the target
                pool = np.delete(np.arange(n), i)
                a, b, c = pop[rng.choice(pool, 3, replace=False)]

                if self.strategy == "best1bin":
                    # pull toward the best: converges faster, explores less
                    base = pop[np.argmin(fitness)]
                    mutant = base + self.F * (b - c)
                else:
                    mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lo, hi)

                # binomial crossover; the forced index guarantees the trial
                # differs from the parent in at least one coordinate, or a low CR
                # would often produce an exact copy and waste the evaluation
                cross = rng.uniform(size=dim) < self.CR
                cross[rng.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])

                # pairwise selection -- see the class docstring
                f_trial = self.func(trial)
                if f_trial <= fitness[i]:
                    pop[i], fitness[i] = trial, f_trial

            self.history_.append(fitness.min())

        best = np.argmin(fitness)
        self.best_ = pop[best]
        self.best_fitness_ = float(fitness[best])
        self.population_ = pop
        return self


class ParticleSwarm:
    """PSO: candidates with momentum, pulled between memory and gossip.

    Each particle has a position and a VELOCITY, and the velocity is what makes
    this different from everything else in the package -- the search has inertia,
    so a particle carries on past a local optimum rather than settling into it.

    The update is three forces::

        v <- w*v  +  c1*r1*(personal_best - x)  +  c2*r2*(global_best - x)
             ^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
             inertia  cognitive: "where I did     social: "where the swarm
                      well"                       did well"

    ``r1`` and ``r2`` are fresh uniform randoms per component, not scalars. That
    detail matters: it makes each dimension pulled slightly differently, which is
    the only source of variety once the swarm starts to gather.

    THE BALANCE, AGAIN
    ------------------
    ``c1`` against ``c2`` is exploration against exploitation in its most literal
    form in this package. All social (c1=0) and every particle rushes the current
    best, and the swarm collapses onto it. All cognitive (c2=0) and the particles
    never share anything -- it becomes N independent hill-climbers.

    ``w`` decays from 0.9 to 0.4 over the run: wide-ranging early, settling late.
    A constant high ``w`` never converges (particles overshoot forever); a
    constant low one converges immediately to a local optimum. The decay is the
    standard remedy, and it is the same annealing idea as in ``annealing.py``.

    Kennedy & Eberhart (1995).
    """

    def __init__(self, func, bounds, n_particles=30, n_iterations=200,
                 w=(0.9, 0.4), c1=1.5, c2=1.5, random_state=None):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        dim = len(self.bounds)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        span = hi - lo

        x = rng.uniform(lo, hi, size=(self.n_particles, dim))
        # velocities start small relative to the search region: a large initial
        # velocity throws particles at the bounds before they have evaluated
        # anything worth moving toward
        v = rng.uniform(-0.1, 0.1, size=(self.n_particles, dim)) * span

        f = np.array([self.func(p) for p in x])
        p_best, p_best_f = x.copy(), f.copy()
        g_idx = np.argmin(f)
        g_best, g_best_f = x[g_idx].copy(), float(f[g_idx])
        self.history_ = [g_best_f]

        w_start, w_end = self.w
        for t in range(self.n_iterations):
            w = w_start + (w_end - w_start) * t / max(self.n_iterations - 1, 1)

            r1 = rng.uniform(size=(self.n_particles, dim))
            r2 = rng.uniform(size=(self.n_particles, dim))
            v = (w * v
                 + self.c1 * r1 * (p_best - x)
                 + self.c2 * r2 * (g_best - x))
            # clamping velocity to the search span stops particles from
            # exploding out of the region faster than the bounds can catch them
            v = np.clip(v, -span, span)
            x = np.clip(x + v, lo, hi)

            f = np.array([self.func(p) for p in x])
            better = f < p_best_f
            p_best[better], p_best_f[better] = x[better], f[better]

            i = np.argmin(p_best_f)
            if p_best_f[i] < g_best_f:
                g_best, g_best_f = p_best[i].copy(), float(p_best_f[i])
            self.history_.append(g_best_f)

        self.best_ = g_best
        self.best_fitness_ = g_best_f
        self.population_ = x
        return self


__all__ = ["DifferentialEvolution", "ParticleSwarm"]
