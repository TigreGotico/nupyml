"""Derivative-free optimization: GAs, CMA-ES, DE, PSO, annealing, NSGA-II, GP.

The tests lean on standard benchmark functions with known optima, because the
claim each method makes is about SEARCH -- and a search either finds the optimum
or it does not.
"""
import numpy as np
import pytest

from nupyml.datasets import make_classification
from nupyml.evolutionary import (
    GeneticAlgorithm, BinaryGeneticAlgorithm, tournament_selection,
    roulette_selection, rank_selection, CMAES, EvolutionStrategy,
    DifferentialEvolution, ParticleSwarm, SimulatedAnnealing, TabuSearch,
    NSGA2, pareto_front, non_dominated_sort, SymbolicRegressor, GeneticProgram,
    GAFeatureSelector, GASearchCV,
)
from nupyml.evolutionary.multiobjective import crowding_distance, _dominates
from nupyml.evolutionary.programming import Node, _protected_div
from nupyml.tree import DecisionTreeClassifier
from nupyml.linear_model import LogisticRegression


# --- benchmark objectives -------------------------------------------------

def sphere(x):
    """Convex, separable, optimum 0 at the origin. If a method cannot do this,
    it cannot do anything."""
    return float((x ** 2).sum())


def rosenbrock(x):
    """A narrow curved valley. The optimum (all ones) is easy to approach and
    hard to reach -- it punishes methods that cannot follow a bend."""
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    """Riddled with local optima on a global bowl. Pure hill climbing dies here."""
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ellipsoid(x):
    """Condition number 1e6: the axes have wildly different scales.

    This is the function CMA-ES exists for, and the one an isotropic search
    cannot do.
    """
    return float(np.sum(1e6 ** (np.arange(len(x)) / max(len(x) - 1, 1)) * x ** 2))


# --- selection operators --------------------------------------------------

def test_tournament_selection_favours_the_good():
    fitness = np.arange(20.0)          # index 0 is best (minimisation)
    picked = tournament_selection(fitness, 2000, k=3, rng=np.random.RandomState(0))
    assert picked.mean() < 8.0         # far below the uniform mean of 9.5


def test_larger_tournaments_apply_more_pressure():
    fitness = np.arange(50.0)
    rng = np.random.RandomState(0)
    gentle = tournament_selection(fitness, 3000, k=2, rng=rng).mean()
    fierce = tournament_selection(fitness, 3000, k=10, rng=rng).mean()
    assert fierce < gentle


def test_tournament_selection_ignores_fitness_magnitude():
    """Rank-based by construction: any monotone rescaling must change nothing."""
    fitness = np.arange(20.0)
    a = tournament_selection(fitness, 500, 3, np.random.RandomState(0))
    b = tournament_selection(fitness ** 3, 500, 3, np.random.RandomState(0))
    assert np.array_equal(a, b)


def test_rank_selection_ignores_fitness_magnitude():
    fitness = np.array([1.0, 2.0, 3.0, 1000.0])
    a = rank_selection(fitness, 500, rng=np.random.RandomState(0))
    b = rank_selection(np.array([1.0, 2.0, 3.0, 4.0]), 500,
                       rng=np.random.RandomState(0))
    assert np.array_equal(a, b)


def test_roulette_selection_is_captured_by_one_dominant_individual():
    """The cautionary tale, demonstrated rather than asserted: roulette reads
    fitness VALUES, so one outlier can take the population. Rank cannot."""
    fitness = np.array([0.0, 999.0, 1000.0, 1000.0])
    rng = np.random.RandomState(0)
    roulette = roulette_selection(fitness, 2000, rng=rng)
    ranked = rank_selection(fitness, 2000, rng=rng)
    assert (roulette == 0).mean() > 0.9       # the best swallows everything
    assert (ranked == 0).mean() < 0.6         # rank stays proportionate


def test_roulette_survives_a_uniform_population():
    """Every individual identical means zero total weight -- a division by zero
    unless it is handled."""
    picked = roulette_selection(np.ones(10), 50, rng=np.random.RandomState(0))
    assert len(picked) == 50


def test_rank_pressure_one_is_uniform():
    picked = rank_selection(np.arange(20.0), 4000, pressure=1.0,
                            rng=np.random.RandomState(0))
    assert picked.mean() == pytest.approx(9.5, abs=0.6)


# --- genetic algorithm ----------------------------------------------------

def test_ga_minimises_sphere():
    ga = GeneticAlgorithm(sphere, [(-5, 5)] * 5, n_generations=100,
                          random_state=0).run()
    assert ga.best_fitness_ < 1e-3


def test_ga_best_never_gets_worse():
    """What elitism buys: a search that cannot move backwards."""
    ga = GeneticAlgorithm(sphere, [(-5, 5)] * 4, n_generations=60, elitism=1,
                          random_state=0).run()
    assert np.all(np.diff(ga.history_) <= 1e-12)


def test_without_elitism_the_best_can_be_lost():
    """The reason elitism is not optional: selection and mutation are free to
    destroy the best individual, so progress is not monotone without it."""
    ga = GeneticAlgorithm(sphere, [(-5, 5)] * 4, n_generations=60, elitism=0,
                          mutation_rate=0.5, random_state=0).run()
    assert np.any(np.diff(ga.history_) > 0)


def test_ga_respects_its_bounds():
    ga = GeneticAlgorithm(sphere, [(1, 2)] * 3, n_generations=20,
                          random_state=0).run()
    assert np.all(ga.population_ >= 1) and np.all(ga.population_ <= 2)


def test_ga_diversity_falls_as_it_converges():
    ga = GeneticAlgorithm(sphere, [(-5, 5)] * 4, n_generations=80,
                          random_state=0).run()
    assert ga.diversity_[-1] < ga.diversity_[0] / 2


def test_extreme_selection_pressure_converges_prematurely():
    """The characteristic GA failure, made to happen: crank the pressure and
    diversity collapses, after which the search is hill climbing from wherever
    it happened to be."""
    kw = dict(bounds=[(-5, 5)] * 6, n_generations=40, mutation_rate=0.0,
              random_state=0)
    fierce = GeneticAlgorithm(rastrigin, tournament_k=25, **kw).run()
    gentle = GeneticAlgorithm(rastrigin, tournament_k=2, **kw).run()
    assert fierce.diversity_[-1] < gentle.diversity_[-1]


def test_ga_selection_methods_all_work():
    for selection in ("tournament", "roulette", "rank"):
        ga = GeneticAlgorithm(sphere, [(-5, 5)] * 3, n_generations=50,
                              selection=selection, random_state=0).run()
        assert ga.best_fitness_ < 1.0


def test_ga_rejects_an_unknown_selection():
    with pytest.raises(ValueError, match="Unknown selection"):
        GeneticAlgorithm(sphere, [(-5, 5)] * 2, selection="nonsense").run()


def test_binary_ga_solves_onemax():
    """Minimise the number of zero bits. The optimum is all ones."""
    ga = BinaryGeneticAlgorithm(lambda x: float((x == 0).sum()), 30,
                                n_generations=100, random_state=0).run()
    assert ga.best_fitness_ == 0.0
    assert ga.best_.all()


def test_binary_ga_output_is_boolean():
    ga = BinaryGeneticAlgorithm(lambda x: float((x == 0).sum()), 10,
                                n_generations=10, random_state=0).run()
    assert ga.best_.dtype == bool


def test_binary_ga_solves_a_deceptive_subset_problem():
    """Only one specific subset is good -- no gradient, no ranking, nothing to
    exploit but the subset score itself."""
    target = np.zeros(20, dtype=bool)
    target[[2, 5, 11, 17]] = True
    ga = BinaryGeneticAlgorithm(
        lambda x: float(np.sum(x.astype(bool) != target)), 20,
        n_generations=120, random_state=0).run()
    assert np.array_equal(ga.best_, target)


# --- evolution strategies -------------------------------------------------

def test_cmaes_minimises_sphere():
    cma = CMAES(sphere, np.ones(5) * 3, sigma=0.5, n_generations=150,
                random_state=0).run()
    assert cma.best_fitness_ < 1e-8


def test_cmaes_solves_rosenbrock():
    """The curved valley: the covariance must align with the bend."""
    cma = CMAES(rosenbrock, np.zeros(4), sigma=0.5, n_generations=600,
                random_state=0).run()
    assert cma.best_fitness_ < 1e-6
    assert np.allclose(cma.best_, np.ones(4), atol=1e-2)


def test_covariance_adaptation_beats_isotropic_search_on_an_ill_conditioned_problem():
    """The reason CMA-ES exists, as a measurement.

    An isotropic gaussian must use one step size for axes whose scales differ by
    1e6, so it is either too big for one or too small for the other. Learning the
    covariance removes the constraint entirely.
    """
    x0 = np.ones(5)
    cma = CMAES(ellipsoid, x0, sigma=0.5, n_generations=500, random_state=0).run()
    es = EvolutionStrategy(ellipsoid, x0, sigma=0.5, n_generations=500,
                           random_state=0).run()
    assert cma.best_fitness_ < es.best_fitness_ / 1e6


def test_cmaes_learns_an_anisotropic_covariance():
    """The covariance should stretch along the cheap axis and flatten across the
    expensive one -- which is the inverse Hessian, learned without derivatives."""
    cma = CMAES(ellipsoid, np.ones(4), sigma=0.5, n_generations=300,
                random_state=0).run()
    scales = np.sqrt(np.diag(cma.covariance_))
    # axis 0 is cheapest, the last is 1e6 times costlier: variance must reflect it
    assert scales[0] > scales[-1] * 100


def test_cmaes_is_invariant_to_monotone_rescaling():
    """Only fitness ORDER is used, so cubing the objective changes nothing."""
    a = CMAES(sphere, np.ones(4), sigma=0.3, n_generations=60, random_state=0).run()
    b = CMAES(lambda x: sphere(x) ** 3, np.ones(4), sigma=0.3, n_generations=60,
              random_state=0).run()
    assert np.allclose(a.best_, b.best_)


def test_cmaes_covariance_stays_symmetric_and_positive_definite():
    cma = CMAES(rosenbrock, np.zeros(3), sigma=0.5, n_generations=200,
                random_state=0).run()
    C = cma.covariance_
    assert np.allclose(C, C.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_evolution_strategy_minimises_sphere():
    es = EvolutionStrategy(sphere, np.ones(4) * 2, sigma=0.5, n_generations=200,
                           random_state=0).run()
    assert es.best_fitness_ < 1e-3


def test_step_size_shrinks_as_it_converges():
    es = EvolutionStrategy(sphere, np.ones(4) * 2, sigma=1.0, n_generations=200,
                           random_state=0).run()
    assert es.sigma_ < 1.0


# --- differential evolution and PSO ---------------------------------------

def test_differential_evolution_minimises_sphere():
    de = DifferentialEvolution(sphere, [(-5, 5)] * 5, n_generations=100,
                               random_state=0).run()
    assert de.best_fitness_ < 1e-4


def test_differential_evolution_handles_multimodal_rastrigin():
    de = DifferentialEvolution(rastrigin, [(-5.12, 5.12)] * 4, n_generations=300,
                               random_state=0).run()
    assert de.best_fitness_ < 1.0


def test_de_never_gets_worse():
    """Pairwise selection makes this automatic -- a trial only ever replaces a
    parent it beat, so no explicit elitism is needed."""
    de = DifferentialEvolution(sphere, [(-5, 5)] * 3, n_generations=50,
                               random_state=0).run()
    assert np.all(np.diff(de.history_) <= 0)


def test_de_best1bin_strategy_also_works():
    de = DifferentialEvolution(sphere, [(-5, 5)] * 4, strategy="best1bin",
                               n_generations=80, random_state=0).run()
    assert de.best_fitness_ < 1e-3


def test_de_respects_bounds():
    de = DifferentialEvolution(sphere, [(2, 3)] * 3, n_generations=30,
                               random_state=0).run()
    assert np.all(de.population_ >= 2) and np.all(de.population_ <= 3)


def test_particle_swarm_minimises_sphere():
    pso = ParticleSwarm(sphere, [(-5, 5)] * 5, n_iterations=150,
                        random_state=0).run()
    assert pso.best_fitness_ < 1e-6


def test_pso_global_best_never_gets_worse():
    pso = ParticleSwarm(sphere, [(-5, 5)] * 4, n_iterations=80,
                        random_state=0).run()
    assert np.all(np.diff(pso.history_) <= 0)


def test_pso_without_social_pull_is_worse_at_sharing():
    """c2=0 removes the swarm entirely: N independent hill climbers that never
    tell each other anything."""
    kw = dict(bounds=[(-5.12, 5.12)] * 6, n_iterations=120, random_state=0)
    social = ParticleSwarm(rastrigin, c1=1.5, c2=1.5, **kw).run()
    solitary = ParticleSwarm(rastrigin, c1=1.5, c2=0.0, **kw).run()
    assert social.best_fitness_ < solitary.best_fitness_


def test_pso_respects_bounds():
    pso = ParticleSwarm(sphere, [(1, 4)] * 3, n_iterations=50,
                        random_state=0).run()
    assert np.all(pso.population_ >= 1) and np.all(pso.population_ <= 4)


# --- annealing and tabu ---------------------------------------------------

def test_simulated_annealing_minimises_sphere():
    sa = SimulatedAnnealing(sphere, np.ones(4) * 2, bounds=[(-5, 5)] * 4,
                            n_iterations=5000, random_state=0).run()
    assert sa.best_fitness_ < 0.5


def test_annealing_accepts_worse_solutions():
    """The whole mechanism. If it never went uphill it would be hill climbing."""
    sa = SimulatedAnnealing(rastrigin, np.ones(3) * 2, bounds=[(-5, 5)] * 3,
                            n_iterations=2000, random_state=0).run()
    assert sa.n_accepted_worse_ > 0


def test_annealing_calibrates_its_temperature_to_the_objective():
    """T is compared against dE, so it is measured in the objective's units.
    A fixed default is a claim about scale, and a wrong one turns the method
    into the hill climbing it exists to avoid -- silently."""
    sa = SimulatedAnnealing(rastrigin, np.ones(3) * 2, bounds=[(-5, 5)] * 3,
                            n_iterations=500, random_state=0).run()
    assert sa.T0_ > 1.0        # rastrigin's uphill moves cost far more than 1


def test_auto_temperature_makes_annealing_scale_invariant():
    """Multiplying the objective by a million must change nothing: T0 scales with
    it, so every acceptance probability is identical."""
    kw = dict(x0=np.ones(3) * 2, bounds=[(-5, 5)] * 3, n_iterations=1500,
              random_state=0)
    plain = SimulatedAnnealing(sphere, **kw).run()
    scaled = SimulatedAnnealing(lambda x: sphere(x) * 1e6, **kw).run()
    assert scaled.T0_ == pytest.approx(plain.T0_ * 1e6, rel=1e-9)
    assert scaled.n_accepted_worse_ == plain.n_accepted_worse_
    assert np.allclose(scaled.best_, plain.best_)


def test_a_fixed_temperature_far_below_the_objective_scale_never_explores():
    """The failure the auto default exists to prevent, shown deliberately."""
    sa = SimulatedAnnealing(rastrigin, np.ones(3) * 2, bounds=[(-5, 5)] * 3,
                            T0=0.001, alpha=1.0, n_iterations=1000,
                            random_state=0).run()
    assert sa.n_accepted_worse_ == 0


def test_annealing_at_zero_temperature_is_hill_climbing():
    """T -> 0 makes exp(-dE/T) vanish: no worsening move is ever accepted."""
    sa = SimulatedAnnealing(rastrigin, np.ones(3) * 2, bounds=[(-5, 5)] * 3,
                            T0=1e-12, T_min=1e-300, alpha=1.0,
                            n_iterations=500, random_state=0).run()
    assert sa.n_accepted_worse_ == 0


def test_annealing_temperature_decays():
    sa = SimulatedAnnealing(sphere, np.ones(2), bounds=[(-5, 5)] * 2,
                            T0=1.0, alpha=0.99, n_iterations=500,
                            random_state=0).run()
    assert sa.temperatures_[-1] < sa.temperatures_[0]
    assert np.all(np.diff(sa.temperatures_) <= 0)


def test_annealing_reports_the_best_seen_not_the_last_visited():
    """The current point wanders uphill, so the two genuinely differ -- reporting
    the final position would throw away the answer."""
    sa = SimulatedAnnealing(rastrigin, np.ones(3) * 3, bounds=[(-5, 5)] * 3,
                            T0=5.0, alpha=0.999, n_iterations=2000,
                            random_state=0).run()
    assert sa.best_fitness_ <= min(sa.history_)


def test_annealing_history_is_monotone():
    sa = SimulatedAnnealing(sphere, np.ones(3), bounds=[(-5, 5)] * 3,
                            n_iterations=1000, random_state=0).run()
    assert np.all(np.diff(sa.history_) <= 0)


def test_tabu_search_minimises_sphere():
    tabu = TabuSearch(sphere, np.ones(4) * 2, [(-5, 5)] * 4, n_iterations=200,
                      random_state=0).run()
    assert tabu.best_fitness_ < 0.5


def test_tabu_history_is_monotone():
    tabu = TabuSearch(sphere, np.ones(3) * 2, [(-5, 5)] * 3, n_iterations=100,
                      random_state=0).run()
    assert np.all(np.diff(tabu.history_) <= 0)


# --- multi-objective ------------------------------------------------------

def test_dominance_is_what_it_says():
    assert _dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert _dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))   # tie + better
    assert not _dominates(np.array([1.0, 3.0]), np.array([3.0, 1.0]))  # trade-off
    assert not _dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))  # equal


def test_non_dominated_sort_layers_correctly():
    objectives = np.array([[1.0, 1.0],     # front 0, dominates everything
                           [2.0, 2.0],     # front 1
                           [3.0, 3.0]])    # front 2
    fronts = non_dominated_sort(objectives)
    assert [list(f) for f in fronts] == [[0], [1], [2]]


def test_a_pure_trade_off_is_all_one_front():
    """Nothing dominates anything: every point is a different compromise."""
    objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    assert len(non_dominated_sort(objectives)) == 1
    assert len(pareto_front(objectives)) == 4


def test_pareto_front_excludes_dominated_points():
    objectives = np.array([[1.0, 4.0], [4.0, 1.0], [5.0, 5.0]])
    assert set(pareto_front(objectives)) == {0, 1}


def test_crowding_distance_keeps_the_extremes():
    """Boundary points get infinity, or the front's range shrinks every
    generation and the extremes are exactly what show each objective's limit."""
    objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    d = crowding_distance(objectives)
    assert np.isinf(d[0]) and np.isinf(d[-1])
    assert np.all(np.isfinite(d[1:-1]))


def test_crowding_distance_prefers_isolation():
    # three points bunched together, one far away
    objectives = np.array([[0.0, 10.0], [1.0, 1.05], [1.0, 1.0], [10.0, 0.0]])
    d = crowding_distance(objectives)
    assert np.all(np.isfinite(d[1:3]))


def test_nsga2_recovers_the_zdt1_front():
    """ZDT1's true front is f2 = 1 - sqrt(f1), which is known analytically -- so
    this measures the answer rather than merely that it ran."""
    def zdt1(x):
        f1 = x[0]
        g = 1 + 9 * x[1:].sum() / (len(x) - 1)
        return np.array([f1, g * (1 - np.sqrt(f1 / g))])

    n = NSGA2(zdt1, [(0, 1)] * 10, population_size=60, n_generations=150,
              random_state=0).run()
    F = n.front_objectives_
    assert np.abs(F[:, 1] - (1 - np.sqrt(F[:, 0]))).mean() < 0.05


def test_nsga2_spreads_along_the_front():
    """Convergence is half the job; without crowding distance the population
    piles onto one easy corner and reports a hundred near-identical answers."""
    def zdt1(x):
        f1 = x[0]
        g = 1 + 9 * x[1:].sum() / (len(x) - 1)
        return np.array([f1, g * (1 - np.sqrt(f1 / g))])

    n = NSGA2(zdt1, [(0, 1)] * 10, population_size=60, n_generations=150,
              random_state=0).run()
    assert n.front_objectives_[:, 0].max() - n.front_objectives_[:, 0].min() > 0.7


def test_nsga2_front_is_internally_non_dominated():
    """The definition, checked directly: nothing on the front may dominate
    anything else on it."""
    def zdt1(x):
        f1 = x[0]
        g = 1 + 9 * x[1:].sum() / (len(x) - 1)
        return np.array([f1, g * (1 - np.sqrt(f1 / g))])

    n = NSGA2(zdt1, [(0, 1)] * 6, population_size=40, n_generations=60,
              random_state=0).run()
    F = n.front_objectives_
    assert not any(_dominates(F[i], F[j])
                   for i in range(len(F)) for j in range(len(F)) if i != j)


def test_nsga2_finds_the_trade_off_between_two_conflicting_targets():
    """Minimise distance to 0 AND to 2: the front is the segment between them,
    and the answer is a range of compromises rather than a point."""
    def conflicting(x):
        return np.array([float(x[0] ** 2), float((x[0] - 2) ** 2)])

    n = NSGA2(conflicting, [(-1, 3)], population_size=40, n_generations=60,
              random_state=0).run()
    xs = n.front_[:, 0]
    assert xs.min() > -0.15 and xs.max() < 2.15
    assert xs.max() - xs.min() > 1.0        # a spread, not a single point


def test_nsga2_population_size_is_held():
    def two(x):
        return np.array([float(x[0] ** 2), float((x[0] - 1) ** 2)])
    n = NSGA2(two, [(-2, 2)], population_size=30, n_generations=20,
              random_state=0).run()
    assert len(n.population_) == 30


# --- genetic programming --------------------------------------------------

def test_protected_division_never_returns_inf():
    """GP builds x/0 constantly; an inf poisons every ancestor's fitness."""
    out = _protected_div(np.array([1.0, 5.0]), np.array([0.0, 1e-15]))
    assert np.all(np.isfinite(out))


def test_node_evaluates_a_whole_column_at_once():
    tree = Node("add", [Node(var=0), Node(value=3.0)])
    X = np.array([[1.0], [2.0]])
    assert np.allclose(tree.evaluate(X), [4.0, 5.0])


def test_node_reports_size_and_depth():
    tree = Node("mul", [Node(var=0), Node("add", [Node(var=1), Node(value=1.0)])])
    assert tree.size() == 5
    assert tree.depth() == 3


def test_node_prints_as_an_expression():
    tree = Node("add", [Node("mul", [Node(var=0), Node(var=1)]), Node(var=2)])
    assert str(tree) == "((x0 * x1) + x2)"


def test_node_copy_is_deep():
    tree = Node("add", [Node(var=0), Node(value=1.0)])
    copy = tree.copy()
    copy.children[0].var = 5
    assert tree.children[0].var == 0


def test_symbolic_regression_recovers_the_generating_equation():
    """The claim that makes GP worth its cost: not a good fit, the actual law."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (200, 3))
    y = X[:, 0] * X[:, 1] + X[:, 2]
    reg = SymbolicRegressor(population_size=500, n_generations=40,
                            random_state=3).fit(X, y)
    assert reg.score(X, y) > 0.99


def test_symbolic_regression_generalises_off_its_training_range():
    """A recovered equation extrapolates; a curve fit does not. This is the
    difference the method is for."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (200, 2))
    y = X[:, 0] * X[:, 1]
    reg = SymbolicRegressor(population_size=500, n_generations=40,
                            random_state=3).fit(X, y)
    far = rng.uniform(5, 8, (50, 2))       # well outside anything it ever saw
    assert reg.score(far, far[:, 0] * far[:, 1]) > 0.9


def test_symbolic_regression_exposes_a_readable_expression():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (100, 2))
    reg = SymbolicRegressor(population_size=100, n_generations=5,
                            random_state=0).fit(X, X[:, 0] + X[:, 1])
    assert isinstance(reg.expression_, str) and reg.expression_
    assert "x0" in str(reg) or "x1" in str(reg)


def test_parsimony_pressure_limits_bloat():
    """Bloat is GP's characteristic failure: neutral code accumulates because it
    is harmless, destroying the readability that is the method's whole point."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (100, 2))
    y = X[:, 0] + X[:, 1]
    lax = SymbolicRegressor(population_size=200, n_generations=25,
                            parsimony_coefficient=0.0, random_state=0).fit(X, y)
    strict = SymbolicRegressor(population_size=200, n_generations=25,
                               parsimony_coefficient=0.05, random_state=0).fit(X, y)
    assert strict.mean_size_[-1] < lax.mean_size_[-1]


def test_symbolic_regression_predicts_finite_values():
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (100, 2))
    reg = SymbolicRegressor(population_size=100, n_generations=10,
                            functions=("add", "sub", "mul", "div"),
                            random_state=0).fit(X, X[:, 0] / (X[:, 1] + 3))
    assert np.all(np.isfinite(reg.predict(rng.uniform(-2, 2, (50, 2)))))


def test_genetic_program_runs_against_an_arbitrary_objective():
    """GP is not only for regression: any scorable tree property will do."""
    gp = GeneticProgram(2, population_size=50, n_generations=5, random_state=0)
    gp.run(lambda tree: abs(tree.size() - 7))    # aim for a 7-node tree
    assert gp.best_.size() == 7


# --- sklearn-facing wrappers ----------------------------------------------

@pytest.fixture(scope="module")
def informative_data():
    return make_classification(n_samples=200, n_features=12, n_informative=4,
                               random_state=0)


def test_ga_feature_selector_drops_noise(informative_data):
    X, y = informative_data
    sel = GAFeatureSelector(DecisionTreeClassifier(max_depth=4, random_state=0),
                            n_generations=15, random_state=0).fit(X, y)
    assert 0 < sel.support_.sum() < 12
    assert sel.transform(X).shape == (len(X), sel.support_.sum())


def test_ga_feature_selector_reports_a_score_not_a_loss(informative_data):
    """The boundary where the sign flips: the GA minimises, sklearn scores
    maximise, and best_score_ must be the latter."""
    X, y = informative_data
    sel = GAFeatureSelector(LogisticRegression(max_iter=200), n_generations=8,
                            random_state=0).fit(X, y)
    assert 0.0 <= sel.best_score_ <= 1.0


def test_ga_feature_selector_never_selects_nothing(informative_data):
    X, y = informative_data
    sel = GAFeatureSelector(LogisticRegression(max_iter=200), n_generations=10,
                            random_state=0).fit(X, y)
    assert sel.support_.any()


def test_ga_feature_selector_predicts_through_the_subset(informative_data):
    X, y = informative_data
    sel = GAFeatureSelector(DecisionTreeClassifier(max_depth=4, random_state=0),
                            n_generations=10, random_state=0).fit(X, y)
    assert sel.predict(X).shape == y.shape


def test_size_penalty_prefers_smaller_subsets(informative_data):
    X, y = informative_data
    kw = dict(estimator=LogisticRegression(max_iter=200), n_generations=15,
              random_state=0)
    lax = GAFeatureSelector(size_penalty=0.0, **kw).fit(X, y)
    strict = GAFeatureSelector(size_penalty=0.05, **kw).fit(X, y)
    assert strict.support_.sum() <= lax.support_.sum()


def test_ga_search_cv_finds_workable_hyperparameters(informative_data):
    X, y = informative_data
    search = GASearchCV(DecisionTreeClassifier(random_state=0),
                        {"max_depth": [1, 2, 3, 5, 8], "min_samples_split": (2, 20)},
                        population_size=10, n_generations=6,
                        random_state=0).fit(X, y)
    assert search.best_params_["max_depth"] in (1, 2, 3, 5, 8)
    assert search.best_score_ > 0.6
    assert search.predict(X).shape == y.shape


def test_ga_search_cv_caches_repeated_evaluations(informative_data):
    """A GA revisits constantly -- elites persist and children resemble parents.
    Without the cache the same expensive fit is paid for again and again."""
    X, y = informative_data
    search = GASearchCV(DecisionTreeClassifier(random_state=0),
                        {"max_depth": [1, 2, 3]}, population_size=10,
                        n_generations=5, random_state=0).fit(X, y)
    assert search.n_evaluations_ <= 3      # only three distinct configurations exist


def test_ga_search_cv_survives_an_invalid_configuration(informative_data):
    """Stochastic search proposes nonsense; that is normal, not an error."""
    X, y = informative_data
    search = GASearchCV(DecisionTreeClassifier(random_state=0),
                        {"max_depth": [0, 1, 2]},   # 0 is invalid
                        population_size=8, n_generations=4,
                        random_state=0).fit(X, y)
    assert search.best_params_["max_depth"] in (1, 2)
