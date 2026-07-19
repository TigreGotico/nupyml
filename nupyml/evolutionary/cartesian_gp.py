"""Genetic programming over an indexed GRAPH of nodes (Miller, 2000)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


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


__all__ = ["CartesianGP"]
