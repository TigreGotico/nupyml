# nupyml documentation

A scikit-learn-like machine learning **and** neural-network library whose only
runtime dependencies are **numpy** and **scipy**. Everything is implemented from
scratch and written to be read: if you can read numpy, you can read every line
of every model here.

These pages are a guided tour. Start with getting started, then read whichever
area you care about — each page explains not just the API but *why* the
algorithms work the way they do.

## Contents

- **[Getting started](getting-started.md)** — install, the two core APIs
  (estimators and autograd), and a first pipeline.
- **[Classic ML](classic-ml.md)** — the scikit-learn-parity estimators: linear
  models, trees and ensembles, SVMs, clustering, decomposition, the works.
- **[Neural networks](neural-networks.md)** — the autograd engine, layers,
  optimizers, and the `MLP*` wrappers.
- **[Statistics & inference](statistics.md)** — the `stats` package (regression
  *with* standard errors, p-values, CIs, diagnostics) and `inference`
  (MCMC, Bayesian optimization, conformal prediction).
- **[Beyond scikit-learn](beyond-sklearn.md)** — the ecosystem-adjacent families:
  graphical models, anomaly/drift, topic models, optimal transport, graphs,
  survival, copulas, recommenders, image features, and more.
- **[The benchmark suite](benchmarks.md)** — `bench/`, a nupyml-only community
  benchmark that doubles as end-to-end QA.
- **[Reading guide](reading-guide.md)** — the handful of files most worth reading
  to understand how the interesting optimizations actually work.
- **[Contributing](contributing.md)** — running the tests, the API contract, and
  how the code is meant to read.

## Design in one paragraph

Every estimator follows the same contract — `fit` / `predict` / `transform`,
parameters set in `__init__` and never mutated there, learned attributes ending
in a trailing underscore — so they compose in a `Pipeline` and are tunable by the
`*SearchCV` estimators. `utils.estimator_checks.check_estimator` enforces that
contract and is worth reading as a statement of what the contract *is*.
Optimizations are not avoided, but they are explained rather than assumed: where
the fast way differs from the obvious way, the code says why.
