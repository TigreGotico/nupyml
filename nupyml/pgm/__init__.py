"""Probabilistic graphical models: reasoning about many variables at once.

THE PROBLEM
-----------
A joint distribution over ``n`` discrete variables has exponentially many
entries -- ten binary variables already need 1024 numbers. You cannot store it,
let alone learn it, and most of those numbers encode relationships that do not
exist. A graphical model exploits the structure: it factorises the joint into
small local pieces according to a GRAPH of direct dependencies, so a sparse
dependency structure becomes a compact, learnable, queryable model.

    P(x1..xn) = product of small factors, one per graph clique/family

Two grammars for the same idea, differing in what the edges mean:

* **Bayesian networks** (``bayes_net.py``) -- a DIRECTED acyclic graph. Each node
  is a variable with a conditional probability table given its parents, and the
  joint is the product of those CPTs. Edges read as "depends directly on", and
  the direction often (not always) reads as causal.
* **Markov random fields** (``markov.py``) -- an UNDIRECTED graph of potentials
  over cliques. No direction, so it suits symmetric relationships (pixels in an
  image, spins in a lattice) where "A causes B" makes no sense.

WHAT YOU DO WITH ONE
--------------------
* **Inference** -- given evidence about some variables, compute the distribution
  of others. Exact methods (``variable_elimination``, belief propagation,
  the junction tree) are in ``inference.py``. This is the hard, interesting part:
  the naive sum over all configurations is exponential, and the whole point of
  the graph is to reorder that sum so it is not.
* **Structure learning** (``structure.py``) -- discover the graph itself from
  data: which variables depend directly on which. Chow-Liu finds the best tree
  in closed form; hill-climbing searches general DAGs by a score.

This closes the gap left when the original roadmap listed "Bayes nets, MRF,
belief propagation, junction tree" and never built them.

Koller & Friedman, *Probabilistic Graphical Models* (2009).
"""
from .bayes_net import BayesianNetwork, DiscreteCPD
from .markov import MarkovRandomField, MarkovChain
from .inference import variable_elimination, belief_propagation
from .structure import chow_liu, hill_climb_structure

__all__ = [
    "BayesianNetwork", "DiscreteCPD", "MarkovRandomField", "MarkovChain",
    "variable_elimination", "belief_propagation", "chow_liu",
    "hill_climb_structure",
]
