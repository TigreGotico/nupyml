"""Inference and uncertainty: sampling, guarantees, and expensive optimization.

Three tools that share a concern -- being honest about what is not known:

* ``mcmc.py`` -- draw samples from a distribution you can only evaluate up to a
  constant. The workhorse of Bayesian inference: it turns "I can compute the
  unnormalised posterior" into "I have samples from the posterior", which is all
  you ever actually need.
* ``conformal.py`` -- wrap ANY model in prediction intervals with a real coverage
  guarantee, assuming almost nothing. The antidote to the learned-uncertainty
  problem that NGBoost and the RVM run into elsewhere in this library.
* ``bayesopt.py`` -- optimize a function that is expensive to evaluate (a
  hyperparameter sweep, a physical experiment) in as few evaluations as possible,
  by modelling it and choosing each next query to be maximally informative.

What unites them is that each MODELS ITS OWN IGNORANCE and acts on it -- MCMC
explores in proportion to probability, conformal reports how wrong it might be,
Bayesian optimization queries where it is most uncertain. That is the thread
worth following through the module.
"""
from .mcmc import MetropolisHastings, GibbsSampler, HamiltonianMC
from .conformal import (ConformalRegressor, ConformalClassifier,
                        MondrianConformalRegressor,
                        ConformalizedQuantileRegression, VennAbersCalibrator,
                        AdaptiveConformalInference)
from .bayesopt import BayesianOptimization, GaussianProcessRegressor
from ._bayesian import MeanFieldVI, SequentialMonteCarlo, ABC, GPLVM
from .aps import APS
from .raps import RAPS
from .jackknife_plus import JackknifePlus
from .enb_pi import EnbPI
from .deep_ensemble import DeepEnsemble
from .nuts import NUTS
from .svgd import SVGD
from .expectation_propagation_classifier import ExpectationPropagationClassifier
from .slice_sampler import SliceSampler
from .relevance_vector_machine import RelevanceVectorMachine
from .bayesian_neural_network import BayesianNeuralNetwork
from .ensemble_kalman_filter import EnsembleKalmanFilter
from .markov_switching_model import MarkovSwitchingModel
from .bayesian_pmf import BayesianPMF
from .rao_blackwellised_particle_filter import RaoBlackwellisedParticleFilter
from .nested_sampling import NestedSampling
from .determinantal_point_process import DeterminantalPointProcess
from .bayesian_quadrature import BayesianQuadrature
from .sparse_variational_gp import SparseVariationalGP

__all__ = [
    "NUTS", "SVGD", "ExpectationPropagationClassifier", "SliceSampler",
    "RelevanceVectorMachine", "BayesianNeuralNetwork",
    "EnsembleKalmanFilter", "MarkovSwitchingModel", "BayesianPMF",
    "RaoBlackwellisedParticleFilter",
    "NestedSampling", "DeterminantalPointProcess", "BayesianQuadrature",
    "SparseVariationalGP",
    "MetropolisHastings", "GibbsSampler", "HamiltonianMC",
    "ConformalRegressor", "ConformalClassifier", "MondrianConformalRegressor",
    "ConformalizedQuantileRegression", "VennAbersCalibrator",
    "AdaptiveConformalInference",
    "BayesianOptimization", "GaussianProcessRegressor",
    "MeanFieldVI", "SequentialMonteCarlo", "ABC", "GPLVM",
    "APS", "RAPS", "JackknifePlus", "EnbPI", "DeepEnsemble",
]
