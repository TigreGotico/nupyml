# The benchmark suite

`bench/` is a small community benchmark with a strict rule: a solution may import
**only nupyml** (plus numpy / scipy / the standard library). It doubles as
end-to-end QA for the library.

## Layout

Each task is a folder with a goal, a dataset loader, and a metric. Every
submission is a single script exposing a `solve(...)` function.

```
bench/
  harness.py            # discovery, import check, subprocess runner, scoreboards
  <task>/
    README.md           # goal, dataset, metric, protocol
    task.py             # KIND, GOAL, METRIC, HIGHER_IS_BETTER, MIN_SCORE, load(), metric()
    submissions/
      baseline_*.py     # shipped baselines; community entries drop <name>.py here
```

The `solve` signature is chosen by the task's `KIND`:

| KIND | signature |
|---|---|
| `supervised` | `solve(X_train, y_train, X_test) -> y_pred` |
| `clustering` | `solve(X) -> labels` |
| `forecast` | `solve(y_history, horizon) -> y_future` |

## Running it

```bash
python bench/harness.py                       # run every task, refresh the boards
python bench/harness.py digits_classification # a single task
```

The harness runs each submission in an **isolated subprocess** with a wall-clock
timeout, after statically checking its imports (an AST parse that rejects
anything outside the allowlist). A crashing, hanging, or rule-breaking entry is
recorded as such and cannot affect the rest of the run — which is exactly what
makes it safe to point at untrusted community submissions. The held-out test
labels are never handed to the submission.

## It doubles as QA

`test/test_bench.py` runs every baseline through the harness and fails the build
if any drops below its task's `MIN_SCORE` floor. So a change that quietly
regresses an estimator is caught by the benchmark, not just by the unit tests.

## Adding an entry

See [`bench/CONTRIBUTING.md`](../bench/CONTRIBUTING.md): drop a `<name>.py` with a
`solve()` into a task's `submissions/`, run the harness, and the scoreboard
re-ranks. Adding a *task* is the `task.py` contract plus a `README.md`.
