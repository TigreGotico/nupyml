"""M5: genetic v5 -- whale optimisation, cooperative coevolution,
surrogate-assisted EA, age-fitness Pareto optimisation.

Each minimiser drives a standard benchmark near its global optimum; the surrogate
does it with far fewer true evaluations than a full search; age-fitness escapes the
many local minima of Rastrigin.
"""
import numpy as np

from nupyml.evolutionary import (WhaleOptimization, CooperativeCoevolution,
                                SurrogateAssistedEA, AgeFitnessParetoOptimization)


def _sphere(x):
    return np.sum(x ** 2)


def _rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def test_whale_optimization_minimises_sphere():
    woa = WhaleOptimization(_sphere, 4, ([-5.0] * 4, [5.0] * 4), n_whales=30,
                            max_iter=150, random_state=0)
    best = woa.optimize()
    assert woa.best_fitness_ < 1e-2
    assert np.allclose(best, 0, atol=0.2)


def test_cooperative_coevolution_solves_separable_problem():
    ce = CooperativeCoevolution(_sphere, 6, ([-5.0] * 6, [5.0] * 6),
                                n_subcomponents=3, pop_size=30, max_iter=80,
                                random_state=0)
    ce.optimize()
    assert ce.best_fitness_ < 0.1


def test_surrogate_assisted_uses_fewer_true_evaluations():
    sa = SurrogateAssistedEA(_sphere, 4, ([-5.0] * 4, [5.0] * 4), pop_size=40,
                             max_iter=25, eval_fraction=0.3, random_state=0)
    sa.optimize()
    assert sa.best_fitness_ < 1.0
    # a full EA would evaluate every candidate every generation
    assert sa.n_true_evals_ < 40 + 25 * 40


def test_age_fitness_pareto_escapes_rastrigin():
    af = AgeFitnessParetoOptimization(_rastrigin, 3, ([-5.0] * 3, [5.0] * 3),
                                      pop_size=50, max_iter=120, random_state=0)
    af.optimize()
    assert af.best_fitness_ < 2.0            # global is 0; naive hillclimb stalls high
