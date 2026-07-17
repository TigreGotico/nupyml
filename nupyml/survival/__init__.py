"""Survival analysis: modelling TIME UNTIL an event, with incomplete data.

THE PROBLEM THAT DEFINES THE FIELD: CENSORING
---------------------------------------------
"How long until X happens?" -- death, failure, churn, relapse. The complication
that makes it its own field is CENSORING: for many subjects the event has not
happened YET when the study ends. A patient alive at the last visit did not live
"exactly 5 years" -- they lived AT LEAST 5 years, and you do not know the true
value.

You cannot throw those subjects away: the ones still event-free are often the
most informative, and dropping them biases every estimate toward shorter times.
And you cannot treat "5 years, censored" as "5 years, died" -- that invents an
event that did not occur. Ordinary regression handles neither. Every method here
exists to use a censored observation for exactly what it says -- a lower bound on
the true time -- and nothing more.

THE TWO QUESTIONS, AND WHO ANSWERS THEM
---------------------------------------
* **What does the survival curve look like?** ``KaplanMeier`` estimates
  ``S(t) = P(survive past t)`` directly from the data, non-parametrically, with
  no covariates. The standard first look at any time-to-event data.
* **What DRIVES the time?** ``CoxPH`` relates covariates to the hazard -- the
  instantaneous risk -- so you can say "this drug halves the risk of the event",
  without ever committing to the shape of the baseline curve. That "without
  committing" is the semi-parametric trick that made Cox regression the most-used
  model in medical statistics.

``NelsonAalen`` estimates the cumulative hazard, the natural companion to
Kaplan-Meier's survival curve.
"""
from .core import KaplanMeier, NelsonAalen, CoxPH
from .advanced import concordance_index, RandomSurvivalForest, AFT

__all__ = ["KaplanMeier", "NelsonAalen", "CoxPH", "concordance_index",
           "RandomSurvivalForest", "AFT"]
