"""Evolve a network's TOPOLOGY, not just its weights (Stanley & Miikkulainen, 2002)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_random_state


class NEAT(BaseEstimator):
    """Evolve a network's TOPOLOGY, not just its weights (Stanley & Miikkulainen, 2002).

    Fixed-topology neuroevolution can only tune weights of a structure you guessed.
    NEAT grows the structure too: genomes start minimal (inputs wired to outputs)
    and mutations ADD connections and ADD nodes (splitting a connection), so
    complexity increases only as it earns fitness. Historical "innovation numbers"
    let genomes of different shapes be compared and crossed, and speciation protects
    new structures long enough to optimise. This is a compact NEAT that evolves
    feed-forward genomes for a supervised task (e.g. XOR).
    """

    def __init__(self, n_inputs, n_outputs, pop_size=150, generations=80,
                 add_conn_rate=0.3, add_node_rate=0.1, weight_mut=0.8,
                 random_state=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.generations = generations
        self.add_conn_rate = add_conn_rate
        self.add_node_rate = add_node_rate
        self.weight_mut = weight_mut
        self.random_state = random_state

    def _new_genome(self, rng):
        # nodes: 0..n_inputs-1 inputs, then bias, then outputs
        n_in = self.n_inputs + 1                          # + bias
        outs = list(range(n_in, n_in + self.n_outputs))
        conns = {}
        for i in range(n_in):
            for o in outs:
                conns[(i, o)] = rng.randn()
        return {"conns": conns, "nodes": n_in + self.n_outputs}

    def _evaluate(self, genome, X):
        n_in = self.n_inputs + 1
        outs = list(range(n_in, n_in + self.n_outputs))
        N = genome["nodes"]
        # topological feed-forward by node index (nodes added keep acyclic order)
        vals = np.zeros((len(X), N))
        vals[:, :self.n_inputs] = X
        vals[:, self.n_inputs] = 1.0                      # bias
        incoming = {}
        for (a, b), w in genome["conns"].items():
            incoming.setdefault(b, []).append((a, w))
        # evaluate non-input nodes in topological order (a hidden node may have a
        # higher index than the output it feeds, so index order is not safe)
        for node in self._topo_order(genome, n_in, N):
            if node in incoming:
                s = sum(w * vals[:, a] for a, w in incoming[node])
                vals[:, node] = np.tanh(s)
        return vals[:, outs]

    @staticmethod
    def _topo_order(genome, n_in, N):
        edges = {}
        indeg = {v: 0 for v in range(N)}
        for (a, b) in genome["conns"]:
            edges.setdefault(a, []).append(b)
            indeg[b] += 1
        ready = [v for v in range(N) if indeg[v] == 0]
        order = []
        while ready:
            v = ready.pop()
            order.append(v)
            for b in edges.get(v, ()):
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
        return [v for v in order if v >= n_in]

    def _mutate(self, genome, rng):
        g = {"conns": dict(genome["conns"]), "nodes": genome["nodes"]}
        for k in g["conns"]:
            if rng.rand() < self.weight_mut:
                g["conns"][k] += rng.randn() * 0.5
        if rng.rand() < self.add_conn_rate:              # add a feed-forward edge
            a = rng.randint(0, g["nodes"]); b = rng.randint(self.n_inputs + 1, g["nodes"])
            if a < b and (a, b) not in g["conns"]:
                g["conns"][(a, b)] = rng.randn()
        if rng.rand() < self.add_node_rate and g["conns"]:  # split an edge
            (a, b) = list(g["conns"])[rng.randint(len(g["conns"]))]
            new = g["nodes"]; g["nodes"] += 1
            w = g["conns"].pop((a, b))
            g["conns"][(a, new)] = 1.0
            g["conns"][(new, b)] = w
        return g

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).reshape(len(X), -1)
        rng = check_random_state(self.random_state)
        pop = [self._new_genome(rng) for _ in range(self.pop_size)]
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            fits = []
            for g in pop:
                pred = self._evaluate(g, X)
                fit = -np.mean((pred - y) ** 2)
                fits.append(fit)
                if fit > best_fit:
                    best_fit, best = fit, {"conns": dict(g["conns"]),
                                           "nodes": g["nodes"]}
            order = np.argsort(fits)[::-1]
            elite = [pop[i] for i in order[:max(2, self.pop_size // 5)]]
            pop = list(elite)
            while len(pop) < self.pop_size:              # reproduce from the elite
                parent = elite[rng.randint(len(elite))]
                pop.append(self._mutate(parent, rng))
        self.best_genome_ = best
        self.best_fitness_ = best_fit
        return self

    def predict(self, X):
        return self._evaluate(self.best_genome_, np.asarray(X, float))


__all__ = ["NEAT"]
