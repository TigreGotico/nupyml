"""Genetic algorithms: the archetype of population search.

THE LOOP
--------
Four steps, repeated::

    evaluate    -- score every individual
    select      -- pick parents, biased toward the good ones     (exploit)
    crossover   -- combine two parents into a child              (recombine)
    mutate      -- perturb the child at random                   (explore)

That is the entire method. Everything else in this module is a choice of how to
do one of those four.

THE PART THAT IS ACTUALLY SUBTLE
--------------------------------
Selection pressure -- how strongly the good individuals are favoured -- is the
one knob that decides whether a GA works.

Too strong and the best individual's descendants take over the population within
a few generations. The population becomes clones of one another, crossover
between identical parents produces the same individual back, and the search
collapses to mutation-only hill-climbing from a single point. This is PREMATURE
CONVERGENCE, and it is the characteristic failure of the method: it looks like
convergence (the best score plateaus, diversity vanishes) but it is stagnation at
whatever the first decent solution happened to be.

Too weak and selection barely biases anything, so the population wanders --
random search with extra steps.

The three selection operators below differ exactly in how they handle this, and
the difference is not cosmetic:

* ``roulette_selection`` -- probability proportional to fitness. Fragile in the
  way that matters: it reads absolute fitness VALUES, so if one individual scores
  1000 and the rest score 1, it takes the entire population immediately. Rescale
  the objective and the algorithm's behaviour changes even though the problem has
  not.
* ``rank_selection`` -- probability from fitness ORDER, ignoring magnitude. The
  gap between first and second is the same whether it is 0.001 or 10^6, so the
  pressure is constant and the runaway above cannot happen.
* ``tournament_selection`` -- pick k at random, take the best. Also rank-based,
  and ``k`` tunes the pressure directly: k=2 is gentle, k=7 is fierce. Cheap
  (no sort, no normalisation), robust, and the usual default.

Notice that two of the three deliberately DISCARD the fitness magnitude. That is
the lesson: the numbers are usually less trustworthy than their order.
"""
import numpy as np

from ..utils import check_random_state


def tournament_selection(fitness, n_select, k=3, rng=None):
    """Pick ``k`` at random, return the best. Repeat.

    Selection pressure rises with ``k``: the winner must beat k-1 others, so a
    larger tournament makes it likelier that only the very best reproduce.

    Note this never looks at fitness values, only compares them -- so it is
    invariant to any monotone rescaling of the objective, which is exactly what
    makes it hard to break.

    OPTIMIZATION: all ``n_select`` tournaments are drawn as one
    ``(n_select, k)`` index matrix and reduced with a single ``argmin`` along the
    rows, rather than looping. Identical result, one vectorized pass.
    """
    rng = check_random_state(rng)
    fitness = np.asarray(fitness)
    entrants = rng.randint(0, len(fitness), size=(n_select, k))
    winners = np.argmin(fitness[entrants], axis=1)
    return entrants[np.arange(n_select), winners]


def roulette_selection(fitness, n_select, rng=None):
    """Probability proportional to fitness. Included mainly as a cautionary tale.

    Since this package minimises, fitness is first flipped (``max - f``) so that
    lower is better. Two problems follow, and both are inherent rather than
    fixable:

    * the flip makes the probabilities depend on the WORST individual, so adding
      one terrible candidate changes everyone else's odds;
    * it reads absolute values, so a single dominant individual can seize the
      whole population in one generation.

    Prefer ``tournament_selection`` or ``rank_selection``.
    """
    rng = check_random_state(rng)
    fitness = np.asarray(fitness, dtype=np.float64)
    weights = fitness.max() - fitness
    total = weights.sum()
    if total <= 0:
        # every individual is identical, so there is nothing to prefer
        return rng.randint(0, len(fitness), size=n_select)
    return rng.choice(len(fitness), size=n_select, p=weights / total)


def rank_selection(fitness, n_select, pressure=1.7, rng=None):
    """Probability from fitness ORDER, not magnitude.

    ``pressure`` in [1, 2] interpolates linearly between uniform selection (1.0,
    no bias at all) and maximum linear bias (2.0, where the worst individual has
    probability zero). Because only the ranking is used, an objective that spans
    six orders of magnitude behaves exactly like one that spans two.
    """
    rng = check_random_state(rng)
    n = len(fitness)
    if n == 1:
        return np.zeros(n_select, dtype=int)
    # rank 0 is the best (lowest fitness); ranks are positions after argsort
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(n)
    # linear ranking: best gets `pressure`, worst gets 2 - pressure
    p = (pressure - (2 * pressure - 2) * ranks / (n - 1)) / n
    return rng.choice(n, size=n_select, p=p / p.sum())


_SELECTORS = {"tournament": tournament_selection,
              "roulette": roulette_selection,
              "rank": rank_selection}


class GeneticAlgorithm:
    """A GA over real-valued vectors.

    Parameters
    ----------
    func : the objective to MINIMISE. Called with a single candidate.
    bounds : (lo, hi) per dimension, as an array of shape (dim, 2).
    elitism : how many of the best survive each generation UNCHANGED.

    WHY ELITISM IS NOT OPTIONAL
    ---------------------------
    Without it a GA can LOSE its best solution: selection is stochastic, and
    crossover and mutation are free to wreck a good individual. It is entirely
    possible for generation 50 to be worse than generation 49 -- a search that
    can move backwards has no guarantee of ever finding anything.

    Copying the best few through untouched makes the best-so-far monotone, which
    is cheap and is why essentially every practical GA does it. The cost is
    selection pressure -- elites also compete as parents -- so keep it small
    (1-2 is standard).
    """

    def __init__(self, func, bounds, population_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.1, mutation_sigma=0.1,
                 selection="tournament", tournament_k=3, elitism=1,
                 random_state=None):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.selection = selection
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.random_state = random_state

    @property
    def dim(self):
        return len(self.bounds)

    def _select(self, fitness, n, rng):
        if self.selection == "tournament":
            return tournament_selection(fitness, n, self.tournament_k, rng)
        if self.selection not in _SELECTORS:
            raise ValueError(f"Unknown selection: {self.selection!r}")
        return _SELECTORS[self.selection](fitness, n, rng=rng)

    def _crossover(self, a, b, rng):
        """Blend crossover (BLX): sample uniformly on the segment joining the
        parents, extended slightly past each end.

        The extension is the point. Sampling strictly BETWEEN the parents means
        the population's spread can only ever shrink -- every child is inside the
        current convex hull, generation after generation, and the search
        contracts to a point regardless of where the optimum is. Reaching a
        little beyond each parent lets the population expand when the optimum
        lies outside it.
        """
        alpha = 0.5  # the reach past each end, as a fraction of the gap
        lo = np.minimum(a, b) - alpha * np.abs(a - b)
        hi = np.maximum(a, b) + alpha * np.abs(a - b)
        return rng.uniform(lo, hi)

    def _mutate(self, x, rng):
        """Gaussian perturbation, applied per gene with probability
        ``mutation_rate``.

        The scale is relative to each dimension's range, so a parameter bounded
        [0, 1] and one bounded [0, 1000] are perturbed comparably. An absolute
        sigma would be a no-op on one and catastrophic on the other.
        """
        mask = rng.uniform(size=self.dim) < self.mutation_rate
        span = self.bounds[:, 1] - self.bounds[:, 0]
        return x + mask * rng.normal(0, self.mutation_sigma * span, size=self.dim)

    def _evaluate(self, population):
        return np.array([self.func(ind) for ind in population])

    def run(self):
        """Returns self, with ``best_``, ``best_fitness_`` and ``history_`` set."""
        rng = check_random_state(self.random_state)
        pop = rng.uniform(self.bounds[:, 0], self.bounds[:, 1],
                          size=(self.population_size, self.dim))
        fitness = self._evaluate(pop)
        self.history_ = [fitness.min()]
        self.diversity_ = [float(pop.std(axis=0).mean())]

        for _ in range(self.n_generations):
            # the elites are carried through untouched -- see the class docstring
            elite_idx = np.argsort(fitness)[:self.elitism]
            children = [pop[i].copy() for i in elite_idx]

            n_children = self.population_size - self.elitism
            parents = self._select(fitness, 2 * n_children, rng)
            for i in range(n_children):
                a, b = pop[parents[2 * i]], pop[parents[2 * i + 1]]
                child = self._crossover(a, b, rng) \
                    if rng.uniform() < self.crossover_rate else a.copy()
                children.append(self._mutate(child, rng))

            pop = np.clip(np.array(children), self.bounds[:, 0], self.bounds[:, 1])
            fitness = self._evaluate(pop)
            self.history_.append(fitness.min())
            # tracked because a collapse to zero IS premature convergence, and
            # the fitness curve alone cannot distinguish that from success
            self.diversity_.append(float(pop.std(axis=0).mean()))

        best = np.argmin(fitness)
        self.best_ = pop[best]
        self.best_fitness_ = float(fitness[best])
        self.population_ = pop
        return self


class BinaryGeneticAlgorithm(GeneticAlgorithm):
    """A GA over bit strings -- the original formulation.

    Worth seeing separately because the binary case is where a GA is least
    replaceable. Real-valued problems usually have a gradient, or CMA-ES, or
    differential evolution, all of which beat a GA. A combinatorial problem --
    which features, which items, which subset -- has none of those, and this is
    the natural representation for it.

    The operators change with the representation, and each is the obvious
    analogue: uniform crossover picks each bit from either parent; mutation flips
    a bit rather than nudging a float.
    """

    def __init__(self, func, n_bits, population_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=None, selection="tournament",
                 tournament_k=3, elitism=1, random_state=None):
        # the classic default: on average one bit per individual flips, which is
        # enough to explore and rarely enough to destroy a good solution
        if mutation_rate is None:
            mutation_rate = 1.0 / n_bits
        super().__init__(func, np.zeros((n_bits, 2)), population_size,
                         n_generations, crossover_rate, mutation_rate,
                         selection=selection, tournament_k=tournament_k,
                         elitism=elitism, random_state=random_state)
        self.n_bits = n_bits

    @property
    def dim(self):
        return self.n_bits

    def _crossover(self, a, b, rng):
        """Uniform crossover: each bit independently from either parent."""
        take_a = rng.uniform(size=self.n_bits) < 0.5
        return np.where(take_a, a, b)

    def _mutate(self, x, rng):
        flip = rng.uniform(size=self.n_bits) < self.mutation_rate
        return np.logical_xor(x.astype(bool), flip).astype(np.float64)

    def run(self):
        rng = check_random_state(self.random_state)
        pop = (rng.uniform(size=(self.population_size, self.n_bits)) < 0.5
               ).astype(np.float64)
        fitness = self._evaluate(pop)
        self.history_ = [fitness.min()]
        self.diversity_ = [float(pop.std(axis=0).mean())]

        for _ in range(self.n_generations):
            elite_idx = np.argsort(fitness)[:self.elitism]
            children = [pop[i].copy() for i in elite_idx]

            n_children = self.population_size - self.elitism
            parents = self._select(fitness, 2 * n_children, rng)
            for i in range(n_children):
                a, b = pop[parents[2 * i]], pop[parents[2 * i + 1]]
                child = self._crossover(a, b, rng) \
                    if rng.uniform() < self.crossover_rate else a.copy()
                children.append(self._mutate(child, rng))

            pop = np.array(children)
            fitness = self._evaluate(pop)
            self.history_.append(fitness.min())
            self.diversity_.append(float(pop.std(axis=0).mean()))

        best = np.argmin(fitness)
        self.best_ = pop[best].astype(bool)
        self.best_fitness_ = float(fitness[best])
        self.population_ = pop
        return self


__all__ = ["GeneticAlgorithm", "BinaryGeneticAlgorithm", "tournament_selection",
           "roulette_selection", "rank_selection"]
