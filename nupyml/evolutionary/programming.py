"""Genetic programming: evolve the FORM of the model, not its parameters.

THE DIFFERENCE THAT MATTERS
---------------------------
Every other learner in this library is handed a form and fits its parameters.
Linear regression is told ``y = w.x + b`` and finds ``w``. A neural network is
told its architecture and finds its weights. The structure is the human's
decision; only the numbers are learned.

Genetic programming evolves the EXPRESSION::

    x0 * x1 + sin(x2)          not     w1*x0 + w2*x1 + w3*x2

The individuals are syntax trees, and crossover swaps subtrees between them. The
search space is not a vector space -- it is the space of programs -- which is
exactly why no gradient exists to follow. There is no derivative with respect to
"replace multiplication with addition".

SYMBOLIC REGRESSION
-------------------
The result is an equation you can read. Fitting Kepler's data with a neural
network gives a black box that predicts orbits; symbolic regression gives back
``T^2 ∝ a^3``. When the goal is to UNDERSTAND rather than predict, an
interpretable form is not a nicety -- it is the entire deliverable, and this is
the only method here that produces one from nothing.

BLOAT: THE CHARACTERISTIC FAILURE
---------------------------------
Trees grow without limit unless stopped. Crossover tends to produce children
larger than their parents, and neutral code -- ``x + 0``, ``x * 1``, a subtree
whose value never matters -- accumulates because it is harmless. It survives
precisely BECAUSE it does nothing: it cannot hurt fitness, so selection never
removes it, and it protects the useful code beneath from disruptive crossover.

Left alone, GP produces a ten-thousand-node expression that computes what five
nodes would, evaluates slowly, and cannot be read -- destroying the one advantage
the method has. The defence is ``parsimony_coefficient``, a penalty on size, and
picking it is a real trade-off: too small and bloat wins, too large and the
search kills complex-but-correct expressions before they are refined.

Koza (1992).
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _protected_div(a, b):
    """Division that cannot produce inf or nan.

    GP will generate ``x / 0`` -- not occasionally, but constantly, since nothing
    stops it from building that subtree. An inf then propagates through every
    ancestor node and poisons the fitness of an individual that may be excellent
    everywhere else. Returning 1.0 for a near-zero denominator keeps the value
    finite and lets selection judge the expression on the rest of its behaviour.

    The cost is honest: this makes the operator discontinuous, so an expression
    can exploit the guard. It is the standard trade in GP because the alternative
    -- nan spreading through the population -- is worse.
    """
    b = np.asarray(b, dtype=np.float64)
    return np.where(np.abs(b) < 1e-9, 1.0, a / np.where(np.abs(b) < 1e-9, 1.0, b))


def _protected_log(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.abs(a) < 1e-9, 0.0, np.log(np.abs(a)))


def _protected_sqrt(a):
    return np.sqrt(np.abs(a))


# (function, arity, name). Arity is what the tree builder needs to know.
FUNCTIONS = {
    "add": (np.add, 2, "+"),
    "sub": (np.subtract, 2, "-"),
    "mul": (np.multiply, 2, "*"),
    "div": (_protected_div, 2, "/"),
    "sin": (np.sin, 1, "sin"),
    "cos": (np.cos, 1, "cos"),
    "log": (_protected_log, 1, "log"),
    "sqrt": (_protected_sqrt, 1, "sqrt"),
    "neg": (np.negative, 1, "neg"),
}


class Node:
    """One node of an expression tree: a function, a variable, or a constant.

    Kept deliberately plain -- the algorithm is the interesting part, and a node
    that is anything more than "an operator and its children" obscures it.
    """

    __slots__ = ("op", "children", "value", "var")

    def __init__(self, op=None, children=None, value=None, var=None):
        self.op = op                      # key into FUNCTIONS, or None for a leaf
        self.children = children or []
        self.value = value                # a constant, if this is one
        self.var = var                    # a feature index, if this is one

    def is_leaf(self):
        return self.op is None

    def evaluate(self, X):
        """Evaluate over a whole dataset at once.

        Every node returns a full COLUMN, not a scalar, so a tree is evaluated
        for all n samples in one recursion rather than n. A leaf broadcasts to
        the right length so its parents need no special case.
        """
        if self.var is not None:
            return X[:, self.var]
        if self.value is not None:
            return np.full(len(X), self.value)
        fn, arity, _ = FUNCTIONS[self.op]
        args = [c.evaluate(X) for c in self.children]
        return fn(*args)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        return 1 + max((c.depth() for c in self.children), default=0)

    def copy(self):
        return Node(self.op, [c.copy() for c in self.children], self.value,
                    self.var)

    def nodes(self):
        """Every node in the tree, for crossover to pick from."""
        yield self
        for c in self.children:
            yield from c.nodes()

    def __str__(self):
        if self.var is not None:
            return f"x{self.var}"
        if self.value is not None:
            return f"{self.value:.3g}"
        _, arity, symbol = FUNCTIONS[self.op]
        if arity == 1:
            return f"{symbol}({self.children[0]})"
        return f"({self.children[0]} {symbol} {self.children[1]})"


class GeneticProgram:
    """Evolve expression trees against an arbitrary fitness function.

    ``fitness_fn(tree)`` returns a number to MINIMISE.
    """

    def __init__(self, n_features, functions=("add", "sub", "mul", "div"),
                 population_size=200, n_generations=30, max_depth=4,
                 tournament_k=3, crossover_rate=0.9, mutation_rate=0.1,
                 parsimony_coefficient=0.001, const_range=(-5.0, 5.0),
                 random_state=None):
        self.n_features = n_features
        self.functions = list(functions)
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.parsimony_coefficient = parsimony_coefficient
        self.const_range = const_range
        self.random_state = random_state

    def _random_leaf(self, rng):
        if rng.uniform() < 0.7:
            return Node(var=rng.randint(self.n_features))
        return Node(value=rng.uniform(*self.const_range))

    def _random_tree(self, rng, depth, method="grow"):
        """Build a tree.

        "full" makes every branch reach ``max_depth``; "grow" stops early at
        random. Ramped half-and-half -- half of each, over a range of depths --
        is the standard initialisation because either method alone produces a
        population of suspiciously similar shapes, and shape diversity is what
        crossover has to work with.
        """
        if depth <= 1 or (method == "grow" and rng.uniform() < 0.3):
            return self._random_leaf(rng)
        op = self.functions[rng.randint(len(self.functions))]
        arity = FUNCTIONS[op][1]
        return Node(op, [self._random_tree(rng, depth - 1, method)
                         for _ in range(arity)])

    def _init_population(self, rng):
        pop = []
        # ramped half-and-half
        for i in range(self.population_size):
            depth = 2 + i % max(self.max_depth - 1, 1)
            method = "full" if i % 2 == 0 else "grow"
            pop.append(self._random_tree(rng, depth, method))
        return pop

    def _crossover(self, a, b, rng):
        """Swap a random subtree of ``a`` for a random subtree of ``b``.

        This is why the representation is a tree: a subtree is a self-contained,
        valid expression, so swapping any two always produces a valid child.
        Crossover on a flat string of tokens would produce syntax errors almost
        every time.
        """
        child = a.copy()
        nodes = list(child.nodes())
        target = nodes[rng.randint(len(nodes))]
        donor = list(b.nodes())[rng.randint(b.size())].copy()
        target.op, target.children = donor.op, donor.children
        target.value, target.var = donor.value, donor.var
        return child

    def _mutate(self, tree, rng):
        """Replace a random subtree with a fresh one."""
        child = tree.copy()
        nodes = list(child.nodes())
        target = nodes[rng.randint(len(nodes))]
        fresh = self._random_tree(rng, max(2, self.max_depth // 2))
        target.op, target.children = fresh.op, fresh.children
        target.value, target.var = fresh.value, fresh.var
        return child

    def _penalised(self, tree, raw):
        return raw + self.parsimony_coefficient * tree.size()

    def run(self, fitness_fn):
        rng = check_random_state(self.random_state)
        pop = self._init_population(rng)

        raw = np.array([fitness_fn(t) for t in pop])
        fitness = np.array([self._penalised(t, r) for t, r in zip(pop, raw)])
        best_i = np.argmin(fitness)
        self.best_ = pop[best_i].copy()
        self.best_fitness_ = float(raw[best_i])
        self.history_ = [self.best_fitness_]
        self.mean_size_ = [float(np.mean([t.size() for t in pop]))]

        for _ in range(self.n_generations):
            # elitism, as in the GA: crossover can wreck the best tree, and a
            # search that loses its best answer has no guarantee at all
            children = [pop[np.argmin(fitness)].copy()]

            while len(children) < self.population_size:
                idx = _tournament(fitness, self.tournament_k, rng)
                if rng.uniform() < self.crossover_rate:
                    mate = _tournament(fitness, self.tournament_k, rng)
                    child = self._crossover(pop[idx], pop[mate], rng)
                elif rng.uniform() < self.mutation_rate:
                    child = self._mutate(pop[idx], rng)
                else:
                    child = pop[idx].copy()

                # a hard depth cap on top of the parsimony penalty: the penalty
                # discourages bloat, this makes it impossible, and without it a
                # single deep tree can make evaluation crawl
                if child.depth() > self.max_depth * 3:
                    child = pop[idx].copy()
                children.append(child)

            pop = children
            raw = np.array([fitness_fn(t) for t in pop])
            fitness = np.array([self._penalised(t, r) for t, r in zip(pop, raw)])
            i = np.argmin(fitness)
            if raw[i] < self.best_fitness_:
                self.best_, self.best_fitness_ = pop[i].copy(), float(raw[i])
            self.history_.append(self.best_fitness_)
            # tracked because runaway growth is the failure mode, and it is
            # invisible in the fitness curve until evaluation slows to a stop
            self.mean_size_.append(float(np.mean([t.size() for t in pop])))

        self.population_ = pop
        return self


def _tournament(fitness, k, rng):
    entrants = rng.randint(0, len(fitness), size=k)
    return entrants[np.argmin(fitness[entrants])]


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """Regression that returns an EQUATION.

    Unlike every other regressor here, the fitted model is a formula you can
    read: print ``expression_`` and you have the relationship, not a weight
    vector. That is the point of the method, and the reason to accept how much
    slower it is than gradient-based fitting.

    Do not expect determinism across ``random_state`` values. GP is a stochastic
    search over a discrete space, so different seeds find genuinely different
    expressions -- sometimes equivalent ones written differently, sometimes worse
    ones. Restarting is a normal part of using it.
    """

    def __init__(self, population_size=500, n_generations=30, max_depth=4,
                 functions=("add", "sub", "mul", "div"), tournament_k=3,
                 crossover_rate=0.9, mutation_rate=0.1,
                 parsimony_coefficient=0.001, const_range=(-5.0, 5.0),
                 random_state=None):
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.functions = functions
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.parsimony_coefficient = parsimony_coefficient
        self.const_range = const_range
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]

        gp = GeneticProgram(
            X.shape[1], self.functions, self.population_size,
            self.n_generations, self.max_depth, self.tournament_k,
            self.crossover_rate, self.mutation_rate,
            self.parsimony_coefficient, self.const_range, self.random_state)

        def mse(tree):
            pred = tree.evaluate(X)
            # a tree can still produce inf despite the protected operators (a
            # tower of multiplications overflows), and an inf fitness is not
            # comparable -- map it to something merely terrible so selection can
            # still rank it against the rest
            if not np.all(np.isfinite(pred)):
                return 1e10
            return float(np.mean((pred - y) ** 2))

        gp.run(mse)
        self.tree_ = gp.best_
        self.expression_ = str(gp.best_)
        self.history_ = gp.history_
        self.mean_size_ = gp.mean_size_
        self._gp = gp
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X)
        pred = self.tree_.evaluate(X)
        return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)

    def __str__(self):
        return getattr(self, "expression_", "SymbolicRegressor (unfitted)")


__all__ = ["SymbolicRegressor", "GeneticProgram", "Node", "FUNCTIONS"]
