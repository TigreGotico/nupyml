"""L5: evolutionary v4 -- GEP, harmony search, memetic algorithm, lexicase, SHADE.

GEP recovers a symbolic formula; harmony search minimises a sphere; the memetic
algorithm and SHADE reach near-optima on the multimodal Rastrigin function; lexicase
selection preserves specialists that each excel at a different case.
"""
import numpy as np
import pytest

from nupyml.evolutionary import (GeneExpressionProgramming, HarmonySearch,
                                 MemeticAlgorithm, lexicase_selection, SHADE)


def _rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def test_gep_recovers_formula():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X.ravel() ** 2 + X.ravel()
    gep = GeneExpressionProgramming(n_inputs=1, head_len=8, pop_size=300,
                                    generations=100, random_state=0).fit(X, y)
    assert gep.best_mse_ < 0.01
    assert np.mean((gep.predict(X) - y) ** 2) < 0.01


def test_harmony_search_minimises_sphere():
    hs = HarmonySearch(lambda x: np.sum(x ** 2), n_var=5, bounds=(-5, 5),
                       iterations=3000, random_state=0).optimize()
    assert np.sum(hs ** 2) < 0.01


def test_memetic_algorithm_near_optimum_on_rastrigin():
    ma = MemeticAlgorithm(_rastrigin, n_var=3, bounds=(-5.12, 5.12),
                          generations=60, random_state=0).optimize()
    assert _rastrigin(ma) < 3.0                           # near the global min (0)


def test_lexicase_preserves_specialists():
    # each individual is best at exactly one case -> lexicase must select all three
    errors = np.array([[0.0, 5, 5], [5, 0.0, 5], [5, 5, 0.0]])
    selected = {lexicase_selection(errors, rng=s) for s in range(30)}
    assert selected == {0, 1, 2}


def test_shade_solves_high_dim_rastrigin():
    sh = SHADE(_rastrigin, n_var=10, bounds=(-5.12, 5.12), pop_size=50,
               generations=300, random_state=0).optimize()
    assert _rastrigin(sh) < 5.0                           # far below the ~90 random baseline
