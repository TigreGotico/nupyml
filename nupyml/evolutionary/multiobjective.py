"""Multi-objective optimization: when "best" is not a thing that exists.

THE REFRAME
-----------
Minimise error AND model size. Maximise accuracy AND fairness. Maximise return
AND minimise risk. These objectives conflict, so there is no single best
solution -- improving one costs the other, and no amount of computation changes
that. It is a property of the problem, not a difficulty to be overcome.

The usual move is to collapse the objectives into one::

    combined = w1 * f1 + w2 * f2

which works and hides the real question. Where did ``w1`` come from? You have
supplied the trade-off as an assumption BEFORE seeing what the trade-offs
actually are. Worse, a weighted sum provably cannot find solutions in a concave
region of the trade-off curve, no matter which weights you pick -- entire
families of good solutions are unreachable, invisibly.

DOMINANCE
---------
``a`` DOMINATES ``b`` if it is at least as good in every objective and strictly
better in at least one. Then ``a`` is better than ``b`` by any reasonable
account, with no weights required.

Most pairs do not dominate each other -- cheaper but less accurate is simply a
different trade-off. Those form the PARETO FRONT: the set where nothing can
improve without something else getting worse. That set is the honest answer to a
multi-objective problem. Pick a point from it afterwards, knowing what each one
costs.

NSGA-II
-------
Evolve the whole front at once. Two ideas do the work:

* **Non-dominated sorting** -- rank by how many layers deep in the front you are.
  This replaces "better fitness" with a partial order, which is the only
  defensible notion of better here.
* **Crowding distance** -- among equally-ranked solutions, prefer the ones in
  sparse regions.

The second is not a tie-break detail; without it the population piles onto
whichever part of the front is easiest to find, and you get a hundred nearly
identical solutions instead of a spread. It is the same diversity problem as
premature convergence in a GA, and the same lesson: the mechanism that keeps
candidates apart is as important as the one that makes them good.

Deb, Pratap, Agarwal & Meyarivan (2002).
"""
import numpy as np

from ..utils import check_random_state


def _dominates(a, b):
    """``a`` is no worse everywhere, and better somewhere. Minimisation."""
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(objectives):
    """Sort into fronts: front 0 is dominated by nothing, front 1 only by front
    0, and so on.

    OPTIMIZATION: the pairwise dominance test is done as a whole-matrix
    comparison rather than an n^2 Python loop. Broadcasting the (n, 1, m) and
    (1, n, m) views gives every pair at once, so the O(n^2 m) work happens in
    numpy instead of the interpreter. The bookkeeping that follows is O(n^2) in
    the worst case but touches only integers.

    Returns a list of arrays of indices, best front first.
    """
    F = np.asarray(objectives, dtype=np.float64)
    n = len(F)
    # dom[i, j] is True when i dominates j
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt

    n_dominated_by = dom.sum(axis=0)      # how many dominate each individual
    fronts = []
    remaining = np.ones(n, dtype=bool)
    counts = n_dominated_by.copy()

    while remaining.any():
        current = np.where(remaining & (counts == 0))[0]
        if len(current) == 0:
            # a cycle cannot happen with a valid dominance relation; if it did,
            # emitting the rest as one front beats looping forever
            fronts.append(np.where(remaining)[0])
            break
        fronts.append(current)
        remaining[current] = False
        # everything the newly-placed front dominated is now one layer freer
        counts = counts - dom[current].sum(axis=0)
    return fronts


def pareto_front(objectives):
    """Indices of the non-dominated solutions -- the trade-off curve itself."""
    return non_dominated_sort(objectives)[0]


def crowding_distance(objectives):
    """How isolated each solution is along the front.

    Per objective, sort and give each point the gap between its neighbours; sum
    over objectives. The boundary points get infinity so they always survive --
    losing them would shrink the front's range every generation, and the extremes
    are exactly the solutions that show what each objective can achieve alone.

    Objectives are normalised by their range first, or one measured in dollars
    would drown one measured in percent.
    """
    F = np.asarray(objectives, dtype=np.float64)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)

    distance = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        span = F[order[-1], j] - F[order[0], j]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span == 0:      # this objective separates nothing here
            continue
        distance[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return distance


class NSGA2:
    """Evolve the Pareto front rather than a single solution.

    ``func`` returns a VECTOR of objectives, all minimised. After ``run``,
    ``front_`` holds the non-dominated solutions and ``front_objectives_`` what
    they cost -- there is no ``best_``, deliberately, because the method's whole
    claim is that no such thing exists.
    """

    def __init__(self, func, bounds, n_objectives=2, population_size=50,
                 n_generations=100, crossover_rate=0.9, mutation_rate=None,
                 eta_c=15.0, eta_m=20.0, random_state=None):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.random_state = random_state

    def _evaluate(self, pop):
        return np.array([np.asarray(self.func(p), dtype=np.float64) for p in pop])

    def _rank_and_crowd(self, objectives):
        """Assign each individual (front index, crowding distance)."""
        n = len(objectives)
        rank = np.zeros(n, dtype=int)
        crowd = np.zeros(n)
        for i, front in enumerate(non_dominated_sort(objectives)):
            rank[front] = i
            crowd[front] = crowding_distance(objectives[front])
        return rank, crowd

    def _tournament(self, rank, crowd, n_select, rng):
        """Crowded tournament: better front wins; ties go to the lonelier point.

        This single comparison is what makes NSGA-II converge to the front AND
        spread along it. Rank alone would converge and clump.
        """
        a = rng.randint(0, len(rank), size=n_select)
        b = rng.randint(0, len(rank), size=n_select)
        a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        return np.where(a_wins, a, b)

    def _sbx(self, p1, p2, rng):
        """Simulated binary crossover: children distributed around the parents
        the way a one-point binary crossover would distribute bit strings.

        ``eta_c`` controls the spread -- large keeps children near the parents.
        """
        u = rng.uniform(size=len(p1))
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return c1, c2

    def _polynomial_mutation(self, x, rng):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        rate = self.mutation_rate if self.mutation_rate is not None \
            else 1.0 / len(x)
        u = rng.uniform(size=len(x))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1 / (self.eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        mask = rng.uniform(size=len(x)) < rate
        return np.clip(x + mask * delta * (hi - lo), lo, hi)

    def run(self):
        rng = check_random_state(self.random_state)
        dim = len(self.bounds)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]

        pop = rng.uniform(lo, hi, size=(self.population_size, dim))
        obj = self._evaluate(pop)

        for _ in range(self.n_generations):
            rank, crowd = self._rank_and_crowd(obj)
            parents = self._tournament(rank, crowd, self.population_size, rng)

            children = []
            for i in range(0, self.population_size, 2):
                p1, p2 = pop[parents[i]], pop[parents[(i + 1) % self.population_size]]
                if rng.uniform() < self.crossover_rate:
                    c1, c2 = self._sbx(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                children.append(self._polynomial_mutation(np.clip(c1, lo, hi), rng))
                if len(children) < self.population_size:
                    children.append(self._polynomial_mutation(np.clip(c2, lo, hi), rng))
            children = np.array(children)

            # elitist survival: parents and children compete TOGETHER, so a good
            # solution can never be lost to a bad generation
            merged = np.vstack([pop, children])
            merged_obj = np.vstack([obj, self._evaluate(children)])
            pop, obj = self._survive(merged, merged_obj)

        front = non_dominated_sort(obj)[0]
        self.population_ = pop
        self.objectives_ = obj
        self.front_ = pop[front]
        self.front_objectives_ = obj[front]
        return self

    def _survive(self, merged, merged_obj):
        """Keep the best population_size: whole fronts, then the least crowded of
        the front that does not fit."""
        selected = []
        for front in non_dominated_sort(merged_obj):
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
                continue
            # this front overflows: take the most isolated members of it, which
            # is where the spread of the final answer is decided
            room = self.population_size - len(selected)
            d = crowding_distance(merged_obj[front])
            selected.extend(front[np.argsort(-d)[:room]])
            break
        selected = np.array(selected, dtype=int)
        return merged[selected], merged_obj[selected]


__all__ = ["NSGA2", "pareto_front", "non_dominated_sort", "crowding_distance"]
