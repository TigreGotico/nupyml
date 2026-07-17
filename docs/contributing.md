# Contributing

## Running the tests

scikit-learn and pandas are **dev-only** dependencies — used to cross-validate
results and to exercise the optional pandas integration, never imported at
runtime.

```bash
pip install -e ".[dev]"
pytest test/
```

All tests live in a single `test/` directory. There is no `importorskip`: a test
that needs a dependency states that dependency in the dev extras, so the suite
runs whole or not at all.

## The API contract

Every estimator follows the same contract, enforced by `check_estimator`:

```python
from nupyml.utils.estimator_checks import check_estimator
check_estimator(MyEstimator())
```

The contract:

- `fit` / `predict` / `transform` as appropriate, and `fit` returns `self`.
- Parameters are set in `__init__` and **never mutated** there — `__init__` only
  stores its arguments verbatim.
- Learned attributes end in a trailing underscore (`coef_`, `labels_`, …).
- `get_params` / `set_params` / `clone` round-trip cleanly, which is what lets an
  estimator drop into a `Pipeline` and be tuned by a `*SearchCV`.

An estimator that passes `check_estimator` composes with everything else in the
library.

## How the code is meant to read

- **Readability first, speed a close second.** Prefer the clear implementation;
  when you optimize, keep the clear version's logic legible and *explain* the
  optimization where it lives — the "why", not just the "what".
- **No hidden compiled extensions.** If you can read numpy, you can read every
  line. Pure numpy/scipy only.
- **Hold each algorithm to the property it advertises.** Tests should check that
  a method *does the thing it claims* (a shrinkage estimator shrinks, a conformal
  interval covers), not merely that it runs. Adversarial cases — malformed input,
  boundary conditions, contract violations — belong in the suite alongside the
  happy path.
- **Match numerical references where they exist.** When a closed form or a
  reference implementation gives the exact answer, test against it; otherwise
  validate on held-out accuracy.

## Benchmarks

The `bench/` suite is both a showcase and a QA gate — see
[the benchmark page](benchmarks.md) and
[`bench/CONTRIBUTING.md`](../bench/CONTRIBUTING.md) for how to add a submission or
a task.
