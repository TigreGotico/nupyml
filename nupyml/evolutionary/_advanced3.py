"""Evolutionary v3: distribution-estimation, strength-Pareto selection, graph
programs, and pack-hunting search.

Four more evolutionary methods. Estimation-of-distribution algorithms (UMDA, cGA)
evolve a PROBABILITY MODEL of good solutions rather than a population. SPEA2 is a
multi-objective EA with a fine-grained fitness that blends dominance and density.
Cartesian genetic programming represents a program as an indexed GRAPH. The grey
wolf optimizer is a swarm metaheuristic modelled on pack hierarchy.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class UMDA(BaseEstimator):
    """Evolve a DISTRIBUTION over good solutions, not a population (Muhlenbein, 1996).

    A genetic algorithm recombines individuals and hopes structure survives. An
    estimation-of-distribution algorithm makes the structure EXPLICIT: it keeps a
    probability model, samples a population from it, selects the best, and re-fits
    the model to them. UMDA is the simplest -- one independent Bernoulli probability
    PER BIT. Each generation, the selected winners' bit-frequencies become the new
    probabilities, so the model marches toward the optimum. It assumes the bits are
    independent, which is exactly its limitation and the reason richer EDAs exist.
    Maximises a binary ``fitness``.
    """

    def __init__(self, n_bits, pop_size=100, select_frac=0.5, generations=100,
                 random_state=None):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.select_frac = select_frac
        self.generations = generations
        self.random_state = random_state

    def optimize(self, fitness):
        rng = check_random_state(self.random_state)
        p = np.full(self.n_bits, 0.5)
        n_sel = max(1, int(self.pop_size * self.select_frac))
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            pop = (rng.rand(self.pop_size, self.n_bits) < p).astype(int)
            fits = np.array([fitness(ind) for ind in pop])
            elite = pop[np.argsort(fits)[::-1][:n_sel]]
            p = np.clip(elite.mean(axis=0), 0.02, 0.98)   # re-fit the model
            if fits.max() > best_fit:
                best_fit = fits.max(); best = pop[fits.argmax()].copy()
        self.prob_ = p
        self.best_, self.best_fitness_ = best, best_fit
        return best


class CompactGA(BaseEstimator):
    """A genetic algorithm compressed to a PROBABILITY VECTOR (Harik, 1999).

    The compact GA is an EDA that mimics a GA with a tiny memory footprint: instead
    of a population it stores one probability per bit. Each step it samples TWO
    individuals, lets them compete, and nudges each bit's probability a small step
    (``1/n``) toward the winner's value -- so over many tournaments the vector drifts
    to the better allele on each bit, reproducing a GA's selection pressure using
    only ``O(n)`` memory. Ideal when a full population will not fit.
    """

    def __init__(self, n_bits, virtual_pop=50, max_iter=5000, random_state=None):
        self.n_bits = n_bits
        self.virtual_pop = virtual_pop
        self.max_iter = max_iter
        self.random_state = random_state

    def optimize(self, fitness):
        rng = check_random_state(self.random_state)
        p = np.full(self.n_bits, 0.5)
        step = 1.0 / self.virtual_pop
        for _ in range(self.max_iter):
            a = (rng.rand(self.n_bits) < p).astype(int)
            b = (rng.rand(self.n_bits) < p).astype(int)
            win, lose = (a, b) if fitness(a) >= fitness(b) else (b, a)
            move = win != lose                            # only bits that differ
            # shift each differing bit 1/N toward the winner's allele
            p[move] += step * (2 * win[move] - 1)
            p = np.clip(p, 0.0, 1.0)
            if np.all((p == 0) | (p == 1)):
                break
        self.prob_ = p
        self.best_ = (p >= 0.5).astype(int)
        self.best_fitness_ = fitness(self.best_)
        return self.best_


class SPEA2(BaseEstimator):
    """Strength-Pareto EA 2: rank by dominance AND density (Zitzler, 2001).

    NSGA-II ranks by non-domination fronts; SPEA2 uses a finer fitness. Each
    solution's STRENGTH is how many others it dominates; its raw fitness is the sum
    of the strengths of everything that dominates IT (so non-dominated solutions
    score 0). A DENSITY term -- the distance to the k-th nearest neighbour -- breaks
    ties toward isolated solutions, spreading the population along the front. An
    external ARCHIVE of the best-so-far is truncated to keep that spread. Minimises a
    vector ``objectives``.
    """

    def __init__(self, objectives, n_var, bounds, pop_size=50, archive_size=50,
                 generations=100, random_state=None):
        self.objectives = objectives
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.generations = generations
        self.random_state = random_state

    def _fitness(self, F):
        n = len(F)
        dom = np.zeros((n, n), bool)
        for i in range(n):
            dom[i] = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        strength = dom.sum(axis=1)
        raw = np.array([strength[dom[:, i]].sum() for i in range(n)])
        from scipy.spatial.distance import cdist
        D = np.sort(cdist(F, F), axis=1)
        k = int(np.sqrt(n))
        density = 1.0 / (D[:, k] + 2.0)
        return raw + density

    def fit(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        archive = np.empty((0, self.n_var))
        for _ in range(self.generations):
            union = np.vstack([pop, archive]) if len(archive) else pop
            F = np.array([self.objectives(x) for x in union])
            fit = self._fitness(F)
            nd = np.where(fit < 1.0)[0]                    # non-dominated
            if len(nd) <= self.archive_size:
                order = np.argsort(fit)[:self.archive_size]
            else:
                order = nd[:self.archive_size]
            archive = union[order]
            # binary tournament + gaussian mutation to make offspring
            offspring = []
            afit = fit[order]
            while len(offspring) < self.pop_size:
                i, j = rng.randint(len(archive), size=2)
                parent = archive[i] if afit[i] <= afit[j] else archive[j]
                child = np.clip(parent + rng.normal(0, 0.1, self.n_var), lo, hi)
                offspring.append(child)
            pop = np.array(offspring)
        self.archive_ = archive
        self.objectives_ = np.array([self.objectives(x) for x in archive])
        return self


class CartesianGP(BaseEstimator):
    """Genetic programming over an indexed GRAPH of nodes (Miller, 2000).

    Tree GP grows and bloats. Cartesian GP fixes the shape: a program is a GRID of
    function nodes, each addressing earlier nodes (or inputs) by INDEX, with a chosen
    output node -- so the genotype is a fixed-length integer string and mutation just
    rewires connections or swaps a function. Nodes not on the path to the output are
    inactive "junk" that can become active by a single mutation, which is a big part
    of why CGP evolves so effectively. Evolved here for symbolic regression with a
    (1+lambda) evolutionary strategy.
    """

    def __init__(self, n_inputs=1, n_nodes=20, generations=300, n_offspring=8,
                 random_state=None):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.generations = generations
        self.n_offspring = n_offspring
        self.random_state = random_state
        self.funcs = [np.add, np.subtract, np.multiply,
                      lambda a, b: a / (np.abs(b) + 1e-6)]

    def _random_genome(self, rng):
        genome = []
        for i in range(self.n_nodes):
            maxref = self.n_inputs + i
            genome.append((rng.randint(len(self.funcs)),
                           rng.randint(maxref), rng.randint(maxref)))
        out = rng.randint(self.n_inputs + self.n_nodes)
        return {"nodes": genome, "out": out}

    def _mutate(self, g, rng):
        ng = {"nodes": [list(n) for n in g["nodes"]], "out": g["out"]}
        for _ in range(2):                                # point mutations
            i = rng.randint(self.n_nodes)
            gene = rng.randint(3)
            maxref = self.n_inputs + i
            ng["nodes"][i][gene] = (rng.randint(len(self.funcs)) if gene == 0
                                    else rng.randint(maxref))
        if rng.rand() < 0.3:
            ng["out"] = rng.randint(self.n_inputs + self.n_nodes)
        return ng

    def _evaluate(self, g, X):
        vals = [X[:, i] for i in range(self.n_inputs)]
        for f, a, b in g["nodes"]:
            vals.append(self.funcs[f](vals[a], vals[b]))
        return vals[g["out"]]

    def fit(self, X, y):
        X = np.atleast_2d(X);
        if X.shape[0] == 1 and X.shape[1] != self.n_inputs:
            X = X.T
        y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        parent = self._random_genome(rng)
        pfit = np.mean((self._evaluate(parent, X) - y) ** 2)
        for _ in range(self.generations):
            for _ in range(self.n_offspring):
                child = self._mutate(parent, rng)
                with np.errstate(all="ignore"):
                    pred = self._evaluate(child, X)
                cfit = np.mean((pred - y) ** 2)
                if np.isfinite(cfit) and cfit <= pfit:    # neutral drift allowed
                    parent, pfit = child, cfit
        self.best_ = parent
        self.best_mse_ = pfit
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] != self.n_inputs:
            X = X.T
        with np.errstate(all="ignore"):
            return self._evaluate(self.best_, X)


class GreyWolfOptimizer(BaseEstimator):
    """Continuous optimisation modelled on wolf-pack hunting (Mirjalili, 2014).

    A swarm metaheuristic with an unusually clean rule. The three best solutions are
    named ALPHA, BETA and DELTA -- the pack leaders -- and every other wolf updates
    its position by moving toward a blend of the three, with a coefficient that
    shrinks over time from EXPLORATION (wolves spread out and search) to
    EXPLOITATION (they converge on the prey). No gradients, few parameters, and it
    is competitive with PSO/DE on multimodal functions. Minimises ``fitness`` over a
    box.
    """

    def __init__(self, fitness, n_var, bounds, n_wolves=20, max_iter=200,
                 random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pos = rng.uniform(lo, hi, (self.n_wolves, self.n_var))
        for it in range(self.max_iter):
            fits = np.array([self.fitness(w) for w in pos])
            order = np.argsort(fits)
            alpha, beta, delta = pos[order[0]], pos[order[1]], pos[order[2]]
            a = 2 - 2 * it / self.max_iter                # explore -> exploit
            new = np.empty_like(pos)
            for i in range(self.n_wolves):
                moves = []
                for leader in (alpha, beta, delta):
                    r1, r2 = rng.rand(self.n_var), rng.rand(self.n_var)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - pos[i])
                    moves.append(leader - A * D)
                new[i] = np.clip(np.mean(moves, axis=0), lo, hi)
            pos = new
        fits = np.array([self.fitness(w) for w in pos])
        self.best_ = pos[fits.argmin()]
        self.best_fitness_ = fits.min()
        return self.best_


__all__ = ["UMDA", "CompactGA", "SPEA2", "CartesianGP", "GreyWolfOptimizer"]
