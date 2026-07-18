"""Quality-diversity and structured evolution: island models, novelty search,
MAP-Elites, and NEAT-lite.

The base package optimises for FITNESS. These change what evolution is FOR:
maintaining diversity (islands), rewarding NOVELTY instead of fitness, filling a
grid of DIVERSE high-performers (MAP-Elites), or evolving network STRUCTURE
(NEAT). All take a fitness (or behaviour) function over real vectors.
"""
import numpy as np

from ..utils import check_random_state


class IslandModelGA:
    """Several sub-populations evolving in parallel, with periodic MIGRATION.

    A single population converges fast -- often prematurely onto a local optimum,
    everyone becoming clones. The island model runs several INDEPENDENT populations
    ("islands"), each exploring its own region, and every so often MIGRATES a few
    best individuals between them. The isolation preserves diversity (different
    islands find different basins); the migration spreads good genes so no island
    is stuck for long. The result explores multimodal landscapes far better than
    one big panmictic population. Maximises ``fitness``.
    """

    def __init__(self, fitness, bounds, n_islands=4, pop_size=20, n_gen=50,
                 migration_interval=10, migration_size=2, mutation=0.1,
                 random_state=None):
        self.fitness = fitness
        self.bounds = bounds
        self.n_islands = n_islands
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.mutation = mutation
        self.random_state = random_state

    def run(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.asarray(self.bounds[0], float), np.asarray(self.bounds[1], float)
        d = len(lo)
        islands = [rng.uniform(lo, hi, (self.pop_size, d))
                   for _ in range(self.n_islands)]
        best, best_f = None, -np.inf
        for g in range(self.n_gen):
            for k, pop in enumerate(islands):
                fits = np.array([self.fitness(x) for x in pop])
                if fits.max() > best_f:
                    best_f, best = fits.max(), pop[fits.argmax()].copy()
                # tournament + mutation within the island
                new = []
                for _ in range(self.pop_size):
                    a, b = rng.randint(self.pop_size, size=2)
                    parent = pop[a] if fits[a] > fits[b] else pop[b]
                    child = parent + self.mutation * rng.randn(d) * (hi - lo)
                    new.append(np.clip(child, lo, hi))
                islands[k] = np.array(new)
            if g % self.migration_interval == 0 and g > 0:
                self._migrate(islands, rng)
        self.best_, self.best_fitness_ = best, best_f
        return self

    def _migrate(self, islands, rng):
        # ring migration: send each island's best few to the next island
        for k in range(self.n_islands):
            src = islands[k]
            fits = np.array([self.fitness(x) for x in src])
            migrants = src[np.argsort(-fits)[:self.migration_size]]
            dst = (k + 1) % self.n_islands
            islands[dst][rng.choice(self.pop_size, self.migration_size,
                                    replace=False)] = migrants


class NoveltySearch:
    """Reward behavioural NOVELTY, not fitness (Lehman & Stanley, 2011).

    On DECEPTIVE problems, climbing the fitness gradient leads straight into a trap
    -- the objective actively misleads. Novelty search abandons the objective: it
    rewards individuals whose BEHAVIOUR is different from everything seen so far
    (mean distance to the k nearest behaviours in an archive). By relentlessly
    seeking new behaviours it explores the whole space and often STUMBLES on the
    goal that direct fitness optimisation never reaches -- the counter-intuitive
    result that "ignoring the objective" can solve it better. ``behavior_fn`` maps
    a genome to a behaviour descriptor.
    """

    def __init__(self, behavior_fn, bounds, pop_size=30, n_gen=50, k=5,
                 mutation=0.15, random_state=None):
        self.behavior_fn = behavior_fn
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.k = k
        self.mutation = mutation
        self.random_state = random_state

    def _novelty(self, beh, archive):
        if len(archive) < self.k:
            return np.inf
        d = np.linalg.norm(np.array(archive) - beh, axis=1)
        return np.mean(np.sort(d)[:self.k])

    def run(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.asarray(self.bounds[0], float), np.asarray(self.bounds[1], float)
        d = len(lo)
        pop = rng.uniform(lo, hi, (self.pop_size, d))
        archive = []
        for _ in range(self.n_gen):
            behs = [np.atleast_1d(self.behavior_fn(x)) for x in pop]
            nov = np.array([self._novelty(b, archive) for b in behs])
            for b in np.array(behs)[np.argsort(-nov)[:max(1, self.pop_size // 6)]]:
                archive.append(b)                      # add the novel ones
            # select by novelty, then mutate
            keep = pop[np.argsort(-nov)[:self.pop_size // 2]]
            children = []
            for _ in range(self.pop_size):
                p = keep[rng.randint(len(keep))]
                children.append(np.clip(p + self.mutation * rng.randn(d) * (hi - lo),
                                        lo, hi))
            pop = np.array(children)
        self.archive_ = np.array(archive)
        self.population_ = pop
        return self


class MAPElites:
    """Illuminate a behaviour space: keep the best individual PER CELL
    (Mouret & Clune, 2015).

    Ordinary optimisation returns ONE solution. MAP-Elites returns a whole map: it
    discretises a BEHAVIOUR space into cells and stores, in each cell, the highest-
    FITNESS individual with that behaviour found so far. So it simultaneously
    optimises (each cell holds an elite) and diversifies (it fills the whole
    behaviour space) -- "quality diversity". The map reveals the fitness-vs-
    behaviour landscape and yields a repertoire of diverse high-performers (many
    ways to walk, not just the fastest). ``descriptor_fn`` maps a genome to
    behaviour coordinates binned into the grid.
    """

    def __init__(self, fitness, descriptor_fn, bounds, descriptor_bounds,
                 grid_shape=(10, 10), n_iter=2000, mutation=0.1, random_state=None):
        self.fitness = fitness
        self.descriptor_fn = descriptor_fn
        self.bounds = bounds
        self.descriptor_bounds = descriptor_bounds
        self.grid_shape = grid_shape
        self.n_iter = n_iter
        self.mutation = mutation
        self.random_state = random_state

    def _cell(self, desc):
        lo, hi = self.descriptor_bounds
        frac = (np.asarray(desc) - lo) / (np.asarray(hi) - lo + 1e-12)
        idx = np.clip((frac * np.array(self.grid_shape)).astype(int),
                      0, np.array(self.grid_shape) - 1)
        return tuple(idx)

    def run(self):
        rng = check_random_state(self.random_state)
        lo, hi = np.asarray(self.bounds[0], float), np.asarray(self.bounds[1], float)
        d = len(lo)
        self.archive_ = {}                             # cell -> (genome, fitness)
        for it in range(self.n_iter):
            if len(self.archive_) < 1:
                x = rng.uniform(lo, hi, d)
            else:                                      # mutate a random elite
                g = list(self.archive_.values())[rng.randint(len(self.archive_))][0]
                x = np.clip(g + self.mutation * rng.randn(d) * (hi - lo), lo, hi)
            f = self.fitness(x)
            cell = self._cell(self.descriptor_fn(x))
            if cell not in self.archive_ or f > self.archive_[cell][1]:
                self.archive_[cell] = (x.copy(), f)    # keep the cell's best
        return self

    def coverage(self):
        return len(self.archive_) / int(np.prod(self.grid_shape))

    def best(self):
        return max(self.archive_.values(), key=lambda v: v[1])


__all__ = ["IslandModelGA", "NoveltySearch", "MAPElites"]
