"""Causal inference: the effect of DOING, not the effect of SEEING.

THE DISTINCTION THAT IS THE WHOLE SUBJECT
-----------------------------------------
Every other model in this library estimates a CORRELATION: given what I observe,
what do I predict? Causal inference asks a different question: if I INTERVENE --
give the treatment, change the price, run the campaign -- what happens?

These are not the same, and conflating them is the costliest mistake in applied
statistics. People who take a drug differ from those who do not in a hundred ways
BESIDES the drug -- sicker people seek treatment, so a naive comparison can make
a helpful drug look harmful. The correlation is real; the causal conclusion drawn
from it is wrong.

THE FUNDAMENTAL PROBLEM
-----------------------
For any one person you see ONE outcome: what happened under the treatment they
actually got. You never see the COUNTERFACTUAL -- what would have happened to that
same person under the other treatment. Causal inference is, at heart, the science
of estimating that missing potential outcome from data, and every method here is
a different assumption about how to fill it in.

THE FIX WHEN YOU CANNOT RANDOMISE
---------------------------------
A randomised trial makes the treated and untreated groups identical on average,
so any outcome difference is the causal effect. When you cannot randomise -- only
observational data -- you must ADJUST for the confounders that differ between
groups:

* ``InversePropensityWeighting`` -- weight subjects by 1/(probability of the
  treatment they got), rebuilding the population a trial would have produced.
* ``DoublyRobust`` -- combine an outcome model AND a propensity model so that if
  EITHER is right, the estimate is unbiased. Two shots at correctness.
* ``propensity_score`` and matching utilities underlie both.

THE ASSUMPTION YOU CANNOT ESCAPE
--------------------------------
Every method here assumes NO UNMEASURED CONFOUNDERS -- that you have adjusted for
everything that affects both treatment and outcome. This is untestable from the
data, and it is where causal claims live or die. The methods are honest bookkeeping
ON TOP of that assumption; they cannot rescue you if it fails. Stating it plainly
is part of using them responsibly.
"""
from .core import (InversePropensityWeighting, DoublyRobust,
                   propensity_score, PropensityMatching, uplift_score)
from .instrumental_variables import InstrumentalVariables
from .double_ml import DoubleML
from .t_learner import TLearner
from .x_learner import XLearner
from .difference_in_differences import difference_in_differences
from .synthetic_control import SyntheticControl
from .notears_linear import notears_linear
from .causal_forest import CausalForest
from .regression_discontinuity import RegressionDiscontinuity
from .r_learner import RLearner

__all__ = ["InversePropensityWeighting", "DoublyRobust", "propensity_score",
           "PropensityMatching", "uplift_score",
           "InstrumentalVariables", "DoubleML", "TLearner", "XLearner",
           "difference_in_differences", "SyntheticControl", "notears_linear",
           "CausalForest", "RegressionDiscontinuity", "RLearner"]
