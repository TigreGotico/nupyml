"""K5: evolutionary v3 -- UMDA, compact GA, SPEA2, Cartesian GP, grey wolf.

The two EDAs solve OneMax by evolving a probability model; SPEA2 sweeps a Pareto
front; Cartesian GP recovers a symbolic formula; the grey wolf optimizer minimises
a continuous benchmark.
"""
import numpy as np
import pytest

from nupyml.evolutionary import (UMDA, CompactGA, SPEA2, CartesianGP,
                                 GreyWolfOptimizer)


def test_umda_solves_onemax():
    best = UMDA(n_bits=20, generations=60, random_state=0).optimize(lambda i: i.sum())
    assert best.sum() >= 19                             # ~ all ones


def test_compact_ga_solves_onemax():
    cga = CompactGA(n_bits=20, virtual_pop=30, max_iter=3000, random_state=0)
    best = cga.optimize(lambda i: i.sum())
    assert best.sum() >= 19
    # the probability vector converged toward 1 on (almost) every bit
    assert (cga.prob_ > 0.5).sum() >= 19


def test_spea2_sweeps_pareto_front():
    def obj(x):
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    sp = SPEA2(obj, n_var=5, bounds=(0, 1), pop_size=50, archive_size=40,
               generations=120, random_state=0).fit()
    F = sp.objectives_
    assert F[:, 0].min() < 0.1 and F[:, 0].max() > 0.9  # spans the front
    err = np.abs(F[:, 1] - (1 - np.sqrt(np.clip(F[:, 0], 0, 1)))).mean()
    assert err < 0.05


def test_cartesian_gp_recovers_formula():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X.ravel() ** 2 + X.ravel()
    cgp = CartesianGP(n_inputs=1, n_nodes=20, generations=400,
                      random_state=0).fit(X, y)
    assert cgp.best_mse_ < 0.01
    assert np.mean((cgp.predict(X) - y) ** 2) < 0.01


def test_grey_wolf_minimises_sphere():
    gw = GreyWolfOptimizer(lambda x: np.sum(x ** 2), n_var=5, bounds=(-5, 5),
                           max_iter=200, random_state=0).optimize()
    assert np.sum(gw ** 2) < 0.01                       # near the global minimum
