"""Evolutionary and derivative-free optimization: search without a gradient.

WHY THIS FAMILY EXISTS
----------------------
Everything else in this library optimizes by asking "which way is downhill?" --
analytically (a normal equation), by autodiff (the tape), or by a numerical
approximation. Gradient descent is enormously efficient when you can use it, and
it is the right default.

But it needs the objective to be differentiable, and plenty of real objectives are
not:

* **Discrete decisions.** "Which subset of features?" has no derivative -- you
  cannot take half a feature.
* **The objective is a black box.** A simulation, a physical experiment, another
  program. You can evaluate it and nothing more.
* **The gradient lies.** Ragged, noisy, or piecewise-constant landscapes have
  gradients that point at the nearest local artefact rather than the solution.
* **The gradient exists but is useless.** Test-set accuracy is piecewise
  constant in the hyperparameters; its derivative is zero almost everywhere.

The methods here need only one thing: the ability to EVALUATE the objective. That
is the weakest possible assumption, and it is what makes them universal.

THE COST OF ASKING FOR SO LITTLE
--------------------------------
A gradient tells you the best direction among infinitely many, in one evaluation.
Without one you must sample directions and compare, so these methods need orders
of magnitude more evaluations. Use them when you must, not when you can
differentiate. A gradient is information; refusing it is expensive.

THE ONE IDEA THEY SHARE
-----------------------
Every method here maintains a POPULATION of candidates, and balances:

* **exploitation** -- look harder near what already works;
* **exploration** -- look somewhere else, in case what works is a local optimum.

All of them are that trade-off with a different mechanism. Simulated annealing
tunes it with a temperature that falls over time. A genetic algorithm gets
exploitation from selection and exploration from mutation. CMA-ES learns the
SHAPE of the region worth exploring. PSO has each particle balance its own best
against the swarm's.

Get the balance wrong in either direction and the method fails predictably: all
exploitation converges instantly to the nearest local optimum; all exploration is
random search.

THE MODULES
-----------
* ``genetic.py``   -- GA: selection, crossover, mutation, elitism. The archetype.
* ``strategies.py``-- CMA-ES and evolution strategies: adapt the search
  distribution itself, which is what makes ES competitive rather than folklore.
* ``swarm.py``     -- differential evolution, PSO. Simple, strong, few knobs.
* ``annealing.py`` -- simulated annealing and tabu search: a single point with a
  principled way to accept a worse one.
* ``multiobjective.py`` -- NSGA-II: when there is no single "best", only
  trade-offs.
* ``programming.py``-- genetic programming and symbolic regression: evolve the
  MODEL, not its parameters.
* ``sklearn_api.py``-- GA-driven feature selection and hyperparameter search.

CONVENTION: EVERYTHING MINIMISES
--------------------------------
Every optimizer here minimises. "Fitness" in the evolutionary literature is
usually maximised, which makes cross-reading a menace, so this package picks one
convention and keeps it -- pass ``-f`` to maximise ``f``. The sklearn-facing
wrappers in ``sklearn_api.py`` are the exception, since sklearn scores are
maximised by convention; they negate internally and say so.
"""
from .genetic import (
    GeneticAlgorithm, BinaryGeneticAlgorithm, tournament_selection,
    roulette_selection, rank_selection,
)
from .strategies import CMAES, EvolutionStrategy
from .swarm import DifferentialEvolution, ParticleSwarm
from .annealing import SimulatedAnnealing, TabuSearch
from .multiobjective import NSGA2, pareto_front, non_dominated_sort
from .programming import SymbolicRegressor, GeneticProgram
from .sklearn_api import GAFeatureSelector, GASearchCV

__all__ = [
    "GeneticAlgorithm", "BinaryGeneticAlgorithm", "tournament_selection",
    "roulette_selection", "rank_selection",
    "CMAES", "EvolutionStrategy",
    "DifferentialEvolution", "ParticleSwarm",
    "SimulatedAnnealing", "TabuSearch",
    "NSGA2", "pareto_front", "non_dominated_sort",
    "SymbolicRegressor", "GeneticProgram",
    "GAFeatureSelector", "GASearchCV",
]
