"""Evolutionary v2: topology evolution, decomposition multi-objective, grammar-
guided programs, and swarm combinatorial search.

Four evolutionary methods beyond the GA/ES/DE/PSO/GP core. NEAT evolves a neural
network's STRUCTURE as well as its weights. MOEA/D solves a multi-objective problem
by DECOMPOSING it into many scalar ones. Grammatical evolution generates programs
that are valid by construction from a GRAMMAR. Ant colony optimisation builds
solutions from a shared PHEROMONE trail.
"""
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


class MOEAD(BaseEstimator):
    """Multi-objective by DECOMPOSITION into scalar subproblems (Zhang & Li, 2007).

    NSGA-II ranks the whole population by dominance. MOEA/D takes a different route:
    spread ``n`` weight vectors over the objective space, turning the problem into
    ``n`` SCALAR subproblems (here via the Tchebycheff aggregation), and evolve them
    together -- each subproblem improved using solutions from its NEIGHBOURS in
    weight space. Because neighbouring weights have similar optima, information
    shares cheaply, and the population sweeps out the Pareto front. ``objectives``
    returns a vector to be minimised.
    """

    def __init__(self, objectives, n_var, bounds, pop_size=50, generations=100,
                 n_neighbors=10, F=0.5, random_state=None):
        self.objectives = objectives
        self.n_var = n_var
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.n_neighbors = n_neighbors
        self.F = F
        self.random_state = random_state

    def fit(self):
        rng = check_random_state(self.random_state)
        lo, hi = self.bounds
        # evenly spaced weight vectors on the 2-objective simplex
        w = np.linspace(0, 1, self.pop_size)
        W = np.column_stack([w, 1 - w]) + 1e-6
        B = np.argsort(np.abs(w[:, None] - w[None, :]), axis=1)[:, :self.n_neighbors]
        pop = rng.uniform(lo, hi, (self.pop_size, self.n_var))
        F = np.array([self.objectives(x) for x in pop])
        z = F.min(axis=0)                                # ideal point
        for _ in range(self.generations):
            for i in range(self.pop_size):
                a, b, c = pop[rng.choice(B[i], 3, replace=True)]
                trial = a + self.F * (b - c)                 # DE-style offspring
                mut = rng.rand(self.n_var) < 1.0 / self.n_var
                trial[mut] += rng.normal(0, 0.1, mut.sum())  # exploration mutation
                trial = np.clip(trial, lo, hi)
                ft = self.objectives(trial)
                z = np.minimum(z, ft)
                for j in B[i]:                            # update neighbours it improves
                    g_new = np.max(W[j] * np.abs(ft - z))       # Tchebycheff
                    g_old = np.max(W[j] * np.abs(F[j] - z))
                    if g_new <= g_old:
                        pop[j] = trial; F[j] = ft
        self.solutions_ = pop
        self.objectives_ = F
        return self


class GrammaticalEvolution(BaseEstimator):
    """Generate programs from a GRAMMAR via an integer genotype (O'Neill & Ryan).

    Genetic programming mutates trees directly and must repair invalid ones.
    Grammatical evolution instead evolves a string of INTEGERS and maps it through a
    BNF grammar: each integer, mod the number of choices for the current rule,
    picks a production. Every genotype therefore decodes to a SYNTACTICALLY VALID
    program, and the search operators stay simple integer mutation/crossover. Used
    here for symbolic regression from a small arithmetic grammar.
    """

    def __init__(self, grammar, fitness, genome_length=40, pop_size=200,
                 generations=60, max_wraps=3, random_state=None):
        self.grammar = grammar                            # dict: symbol -> list of productions
        self.fitness = fitness                            # callable(expr_string) -> score (higher better)
        self.genome_length = genome_length
        self.pop_size = pop_size
        self.generations = generations
        self.max_wraps = max_wraps
        self.random_state = random_state

    def decode(self, genome, start="expr"):
        out = []
        stack = [start]
        i = 0
        steps = 0
        limit = len(genome) * self.max_wraps
        while stack and steps < limit:
            sym = stack.pop()
            if sym not in self.grammar:
                out.append(sym); continue
            prods = self.grammar[sym]
            choice = prods[genome[i % len(genome)] % len(prods)]
            i += 1; steps += 1
            stack.extend(reversed(choice))
        return "".join(out)

    def fit(self):
        rng = check_random_state(self.random_state)
        pop = rng.randint(0, 256, (self.pop_size, self.genome_length))
        best, best_fit = None, -np.inf
        for _ in range(self.generations):
            fits = np.array([self.fitness(self.decode(g)) for g in pop])
            gi = fits.argmax()
            if fits[gi] > best_fit:
                best_fit, best = fits[gi], pop[gi].copy()
            order = fits.argsort()[::-1]
            elite = pop[order[:self.pop_size // 5]]
            new = [elite[i % len(elite)].copy() for i in range(2)]
            while len(new) < self.pop_size:
                p1, p2 = elite[rng.randint(len(elite))], elite[rng.randint(len(elite))]
                cut = rng.randint(1, self.genome_length)
                child = np.concatenate([p1[:cut], p2[cut:]])
                mask = rng.rand(self.genome_length) < 0.1
                child[mask] = rng.randint(0, 256, mask.sum())
                new.append(child)
            pop = np.array(new)
        self.best_genome_ = best
        self.best_expression_ = self.decode(best)
        self.best_fitness_ = best_fit
        return self


class AntColonyOptimization(BaseEstimator):
    """Build tours from a shared PHEROMONE trail (Dorigo, 1992).

    Ants find short paths without a map: each lays PHEROMONE on the route it takes,
    shorter routes get reinforced faster (less evaporation time), and later ants
    prefer strong trails -- a positive feedback that converges on good solutions.
    ACO copies this for combinatorial problems like the TSP: each iteration, ants
    build tours choosing the next city by pheromone^alpha * (1/distance)^beta, then
    deposit pheromone inversely proportional to tour length, with evaporation to
    forget bad early choices. Returns the best tour found.
    """

    def __init__(self, n_ants=20, alpha=1.0, beta=3.0, evaporation=0.5,
                 iterations=100, random_state=None):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, distances):
        D = np.asarray(distances, float)
        n = len(D)
        rng = check_random_state(self.random_state)
        tau = np.ones((n, n))                             # pheromone
        eta = 1.0 / (D + np.eye(n) * 1e9)                 # desirability = 1/distance
        best_tour, best_len = None, np.inf
        for _ in range(self.iterations):
            all_tours, all_lens = [], []
            for _ant in range(self.n_ants):
                start = rng.randint(n)
                tour = [start]; unvisited = set(range(n)) - {start}
                while unvisited:
                    cur = tour[-1]
                    cities = list(unvisited)
                    w = (tau[cur, cities] ** self.alpha) * (eta[cur, cities] ** self.beta)
                    w = w / w.sum()
                    nxt = cities[rng.choice(len(cities), p=w)]
                    tour.append(nxt); unvisited.discard(nxt)
                length = sum(D[tour[i], tour[i + 1]] for i in range(n - 1)) + D[tour[-1], tour[0]]
                all_tours.append(tour); all_lens.append(length)
                if length < best_len:
                    best_len, best_tour = length, tour
            tau *= (1 - self.evaporation)                 # evaporate
            for tour, length in zip(all_tours, all_lens):
                for i in range(n):                        # deposit on used edges
                    a, b = tour[i], tour[(i + 1) % n]
                    tau[a, b] += 1.0 / length; tau[b, a] += 1.0 / length
        self.best_tour_ = best_tour
        self.best_length_ = best_len
        return self


__all__ = ["NEAT", "MOEAD", "GrammaticalEvolution", "AntColonyOptimization"]
