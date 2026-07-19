"""Programs as fixed-length LINEAR genes that are always valid (Ferreira, 2001)."""
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


__all__ = ["GeneExpressionProgramming"]
