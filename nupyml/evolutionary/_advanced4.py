"""Evolutionary v4: linear-genome programs, harmony search, memetic refinement,
case-by-case selection, and adaptive differential evolution.

Five more evolutionary methods. Gene expression programming encodes trees as fixed
LINEAR strings that are always valid. Harmony search improvises solutions like a
musician. The memetic algorithm bolts LOCAL SEARCH onto a GA. Lexicase selection
picks parents case by case, preserving specialists. SHADE adapts its own control
parameters from a history of what worked.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class GeneExpressionProgramming(BaseEstimator):
    """Programs as fixed-length LINEAR genes that are always valid (Ferreira, 2001).

    Tree-based genetic programming must repair the broken trees crossover produces.
    Gene expression programming separates genotype from phenotype: the genome is a
    fixed-length STRING with a ``head`` (functions or terminals) and a ``tail`` (only
    terminals), sized so that decoding it breadth-first (Karva notation) ALWAYS yields
    a syntactically valid expression tree -- no matter how it is mutated or crossed.
    So the search operators stay simple string edits while every individual is a legal
    program. Evolved here for symbolic regression.
    """

    def __init__(self, n_inputs=1, head_len=8, pop_size=200, generations=100,
                 mutation_rate=0.1, random_state=None):
        self.n_inputs = n_inputs
        self.head_len = head_len
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.funcs = {"+": (np.add, 2), "-": (np.subtract, 2), "*": (np.multiply, 2),
                      "/": (lambda a, b: a / (np.abs(b) + 1e-6), 2)}
        self.func_syms = list(self.funcs)

    def _terminals(self):
        return [f"x{i}" for i in range(self.n_inputs)] + ["1"]

    def _gene_len(self):
        return self.head_len + self.head_len + 1          # tail = head*(2-1)+1

    def _random_gene(self, rng):
        terms = self._terminals()
        head = [rng.choice(self.func_syms + terms) for _ in range(self.head_len)]
        tail = [rng.choice(terms) for _ in range(self.head_len + 1)]
        return head + tail

    def _evaluate(self, gene, X):
        # decode Karva: breadth-first, allocating children level by level
        vals = {f"x{i}": X[:, i] for i in range(self.n_inputs)}
        vals["1"] = np.ones(len(X))
        queue = [0]; children_ptr = 1
        # first pass: determine each symbol's argument indices
        args = {}
        i = 0
        frontier = [0]
        used = 1
        while frontier:
            nxt = []
            for pos in frontier:
                sym = gene[pos]
                if sym in self.funcs:
                    ar = self.funcs[sym][1]
                    args[pos] = list(range(used, used + ar))
                    nxt.extend(args[pos]); used += ar
            frontier = nxt

        def ev(pos):
            sym = gene[pos]
            if sym in self.funcs:
                fn = self.funcs[sym][0]
                a = [ev(c) for c in args[pos]]
                return fn(*a)
            return vals[sym]
        with np.errstate(all="ignore"):
            return ev(0)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] != self.n_inputs:
            X = X.T
        y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        pop = [self._random_gene(rng) for _ in range(self.pop_size)]
        best, best_mse = None, np.inf
        for _ in range(self.generations):
            fits = []
            for g in pop:
                with np.errstate(all="ignore"):
                    pred = self._evaluate(g, X)
                mse = np.mean((pred - y) ** 2) if np.all(np.isfinite(pred)) else 1e18
                fits.append(mse)
                if mse < best_mse:
                    best_mse, best = mse, list(g)
            order = np.argsort(fits)
            elite = [pop[i] for i in order[:self.pop_size // 5]]
            new = [list(best)]
            terms = self._terminals()
            while len(new) < self.pop_size:
                p = list(elite[rng.randint(len(elite))])
                for k in range(len(p)):                    # mutation (head/tail aware)
                    if rng.rand() < self.mutation_rate:
                        if k < self.head_len:
                            p[k] = rng.choice(self.func_syms + terms)
                        else:
                            p[k] = rng.choice(terms)
                new.append(p)
            pop = new
        self.best_gene_, self.best_mse_ = best, best_mse
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] != self.n_inputs:
            X = X.T
        with np.errstate(all="ignore"):
            return self._evaluate(self.best_gene_, X)


class HarmonySearch(BaseEstimator):
    """Improvise solutions like a musician tuning a chord (Geem et al., 2001).

    Harmony search draws its metaphor from a band searching for a pleasing harmony.
    It keeps a MEMORY of good solutions, and each new candidate is composed
    note-by-note (dimension by dimension): with high probability take that note from
    a remembered solution (and maybe nudge its PITCH slightly), otherwise play a
    random note. If the new harmony is better than the worst in memory, it replaces
    it. The mix of memory, small adjustments, and occasional randomness balances
    exploitation and exploration with very few parameters. Minimises ``fitness`` over
    a box.
    """

    def __init__(self, fitness, n_var, bounds, memory_size=30, hmcr=0.9, par=0.3,
                 bandwidth=0.05, iterations=2000, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.memory_size = memory_size
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        self.iterations = iterations
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        HM = rng.uniform(lo, hi, (self.memory_size, self.n_var))
        fit = np.array([self.fitness(h) for h in HM])
        span = hi - lo
        for _ in range(self.iterations):
            new = np.empty(self.n_var)
            for j in range(self.n_var):
                if rng.rand() < self.hmcr:                 # memory consideration
                    new[j] = HM[rng.randint(self.memory_size), j]
                    if rng.rand() < self.par:              # pitch adjustment
                        new[j] += rng.uniform(-1, 1) * self.bandwidth * span
                else:
                    new[j] = rng.uniform(lo, hi)           # random note
            new = np.clip(new, lo, hi)
            f = self.fitness(new)
            worst = fit.argmax()
            if f < fit[worst]:                             # replace the worst harmony
                HM[worst] = new; fit[worst] = f
        self.best_ = HM[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


class MemeticAlgorithm(BaseEstimator):
    """A GA that also does LOCAL SEARCH (Moscato, 1989).

    A genetic algorithm explores globally but converges slowly to a precise optimum;
    local search refines precisely but gets stuck. A memetic algorithm combines them:
    after the usual selection/crossover/mutation, every offspring is polished by a few
    steps of LOCAL SEARCH before it competes -- Lamarckian, since the improved genotype
    is kept and inherited. The GA supplies exploration and the local search supplies
    exploitation, so it reaches better optima in fewer generations than either alone.
    Minimises ``fitness`` over a box.
    """

    def __init__(self, fitness, n_var, bounds, pop_size=30, generations=50,
                 local_steps=10, local_sigma=0.1, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.local_steps = local_steps
        self.local_sigma = local_sigma
        self.random_state = random_state

    def _local(self, x, rng):
        fx = self.fitness(x)
        for _ in range(self.local_steps):                  # random-restart hill climb
            cand = np.clip(x + rng.normal(0, self.local_sigma, self.n_var),
                           *self.bounds)
            fc = self.fitness(cand)
            if fc < fx:
                x, fx = cand, fc
        return x, fx

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        fit = np.empty(self.pop_size)
        for i in range(self.pop_size):
            pop[i], fit[i] = self._local(pop[i], rng)
        for _ in range(self.generations):
            order = np.argsort(fit)
            elite = pop[order[:max(2, self.pop_size // 4)]]
            new, newfit = list(elite), list(fit[order[:len(elite)]])
            while len(new) < self.pop_size:
                a, b = elite[rng.randint(len(elite))], elite[rng.randint(len(elite))]
                mask = rng.rand(self.n_var) < 0.5
                child = np.where(mask, a, b)               # uniform crossover
                child = np.clip(child + rng.normal(0, 0.1, self.n_var), lo, hi)
                child, fc = self._local(child, rng)        # Lamarckian refinement
                new.append(child); newfit.append(fc)
            pop, fit = np.array(new), np.array(newfit)
        self.best_ = pop[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


def lexicase_selection(errors, rng=None):
    """Select a parent CASE BY CASE, keeping specialists (Spector, 2012).

    Averaging performance across training cases rewards jack-of-all-trades and
    quietly eliminates individuals that are the ONLY ones solving some hard case.
    Lexicase selection avoids the average entirely: shuffle the cases, then filter the
    candidate pool one case at a time, keeping only those with the BEST error on the
    current case, until a single individual remains. Because the case order is random
    each selection, specialists on rare-but-important cases regularly survive, which
    preserves the diversity that keeps evolution from stalling. ``errors`` is an
    (n_individuals, n_cases) array (lower = better); returns the selected index.
    """
    rng = rng if rng is not None else np.random.RandomState()
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    errors = np.asarray(errors, float)
    candidates = np.arange(len(errors))
    cases = rng.permutation(errors.shape[1])
    for c in cases:
        best = errors[candidates, c].min()
        candidates = candidates[errors[candidates, c] <= best + 1e-12]
        if len(candidates) == 1:
            break
    return int(rng.choice(candidates))


class SHADE(BaseEstimator):
    """Differential evolution that ADAPTS its own knobs (Tanabe & Fukunaga, 2013).

    Differential evolution's performance hinges on two constants -- the scale ``F`` and
    crossover rate ``CR`` -- and the best values change during the run and across
    problems. SHADE removes the guesswork with SUCCESS-HISTORY ADAPTATION: it keeps a
    memory of the ``F``/``CR`` values that recently produced IMPROVEMENTS, samples new
    trials' parameters from that memory, and updates it toward what keeps working
    (weighted Lehmer mean). Combined with current-to-pbest mutation and an archive of
    replaced solutions, it is a top-tier black-box optimiser with essentially no tuning.
    Minimises ``fitness``.
    """

    def __init__(self, fitness, n_var, bounds, pop_size=50, generations=200,
                 memory_size=10, p_best=0.1, random_state=None):
        self.fitness = fitness
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.memory_size = memory_size
        self.p_best = p_best
        self.random_state = random_state

    def optimize(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        fit = np.array([self.fitness(x) for x in pop])
        MF = np.full(self.memory_size, 0.5)
        MCR = np.full(self.memory_size, 0.5)
        mem_pos = 0
        archive = []
        for _ in range(self.generations):
            sF, sCR, weights = [], [], []
            n_best = max(2, int(self.p_best * self.pop_size))
            best_idx = np.argsort(fit)[:n_best]
            for i in range(self.pop_size):
                r = rng.randint(self.memory_size)
                CR = np.clip(rng.normal(MCR[r], 0.1), 0, 1)
                F = -1
                while F <= 0:
                    F = min(_cauchy(rng, MF[r], 0.1), 1.0)
                pbest = pop[rng.choice(best_idx)]
                a = pop[rng.randint(self.pop_size)]
                pool = np.vstack([pop] + ([np.array(archive)] if archive else []))
                b = pool[rng.randint(len(pool))]
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)
                cross = rng.rand(self.n_var) < CR
                cross[rng.randint(self.n_var)] = True
                trial = np.clip(np.where(cross, mutant, pop[i]), lo, hi)
                ft = self.fitness(trial)
                if ft < fit[i]:                            # success -> record F, CR
                    archive.append(pop[i].copy())
                    sF.append(F); sCR.append(CR); weights.append(fit[i] - ft)
                    pop[i] = trial; fit[i] = ft
            if len(archive) > self.pop_size:
                archive = [archive[k] for k in rng.choice(len(archive),
                                                          self.pop_size, replace=False)]
            if sF:                                          # update the memories
                w = np.array(weights); w /= w.sum()
                MF[mem_pos] = np.sum(w * np.array(sF) ** 2) / np.sum(w * np.array(sF))
                MCR[mem_pos] = np.sum(w * np.array(sCR) ** 2) / (
                    np.sum(w * np.array(sCR)) + 1e-12)
                mem_pos = (mem_pos + 1) % self.memory_size
        self.best_ = pop[fit.argmin()]; self.best_fitness_ = fit.min()
        return self.best_


def _cauchy(rng, loc, scale):
    return loc + scale * np.tan(np.pi * (rng.rand() - 0.5))


__all__ = ["GeneExpressionProgramming", "HarmonySearch", "MemeticAlgorithm",
           "lexicase_selection", "SHADE"]
