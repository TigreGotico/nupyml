"""J5: evolutionary v2 -- NEAT, MOEA/D, grammatical evolution, ant colony.

NEAT must evolve a topology that solves XOR (a problem needing a hidden unit);
MOEA/D must sweep out a Pareto front; grammatical evolution must find an exact
symbolic expression; ant colony must beat random tours on a small TSP.
"""
import numpy as np
import pytest

from nupyml.evolutionary import (NEAT, MOEAD, GrammaticalEvolution,
                                 AntColonyOptimization)


def test_neat_evolves_topology_to_solve_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0], float)
    # NEAT is stochastic; restarts are standard practice -- take the best of a few
    solved = False
    for seed in range(5):
        neat = NEAT(2, 1, pop_size=200, generations=250, add_node_rate=0.25,
                    random_state=seed).fit(X, y)
        pred = (neat.predict(X).ravel() > 0.5).astype(int)
        if (pred == y).all():
            solved = True
            # solving XOR requires structure beyond the initial input->output wiring
            assert neat.best_genome_["nodes"] >= 4
            break
    assert solved


def test_moead_sweeps_pareto_front():
    def obj(x):                                           # ZDT1-style convex front
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    m = MOEAD(obj, n_var=5, bounds=(0, 1), pop_size=60, generations=250,
              random_state=0).fit()
    F = m.objectives_
    assert F[:, 0].min() < 0.1 and F[:, 0].max() > 0.9   # spans the front
    # solutions lie on the true front f2 = 1 - sqrt(f1)
    err = np.abs(F[:, 1] - (1 - np.sqrt(np.clip(F[:, 0], 0, 1)))).mean()
    assert err < 0.05


def test_grammatical_evolution_finds_expression():
    grammar = {
        'expr': [['expr', 'op', 'expr'], ['(', 'expr', 'op', 'expr', ')'],
                 ['var'], ['var']],
        'op': [['+'], ['*'], ['-']],
        'var': [['x'], ['x']],
    }
    xs = np.linspace(-2, 2, 20)
    target = xs ** 2 + xs

    def fitness(expr):
        try:
            pred = np.array([eval(expr, {'x': xi, '__builtins__': {}}) for xi in xs])
            if not np.all(np.isfinite(pred)):
                return -1e9
            return -np.mean((pred - target) ** 2)
        except Exception:
            return -1e9

    ge = GrammaticalEvolution(grammar, fitness, genome_length=30, pop_size=300,
                              generations=80, random_state=0).fit()
    assert ge.best_fitness_ > -1e-6                       # essentially exact fit
    # the decoded program is syntactically valid and reproduces the target
    pred = np.array([eval(ge.best_expression_, {'x': xi, '__builtins__': {}})
                     for xi in xs])
    assert np.allclose(pred, target, atol=1e-6)


def test_ant_colony_beats_random_tours():
    rng = np.random.RandomState(0)
    pts = rng.rand(8, 2)
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    aco = AntColonyOptimization(n_ants=20, iterations=100, random_state=0).fit(D)
    rand_lengths = [sum(D[t[i], t[(i + 1) % 8]] for i in range(8))
                    for t in [rng.permutation(8) for _ in range(2000)]]
    assert aco.best_length_ <= min(rand_lengths)          # no random tour is shorter
    assert sorted(aco.best_tour_) == list(range(8))       # a valid permutation
