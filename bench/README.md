# nupyml benchmark suite

A set of small, self-contained ML tasks with a strict rule: **you may only use
`nupyml`** (plus `numpy`, `scipy`, and the standard library) to solve them. No
scikit-learn, no XGBoost, no PyTorch, no pandas — if you want a gradient-boosted
model or a transformer, you build it from what nupyml gives you.

It serves two purposes at once:

1. **A benchmark & learning ground.** Each task is a folder with a goal, a
   dataset, and a metric. Anyone can drop in a single-file solution and see where
   it lands on the scoreboard. Because the toolbox is fixed, the board measures
   *how well you used the algorithms*, not which library you reached for.
2. **A QA harness.** The baselines here exercise dozens of nupyml estimators
   end-to-end on real tasks. A change that quietly breaks accuracy shows up as a
   scoreboard drop, and `test/test_bench.py` fails the build if any baseline
   falls below its task's floor.

## Layout

```
bench/
  harness.py            # runs submissions, scores them, writes the scoreboards
  _run_one.py           # runs one submission in an isolated subprocess
  <task>/
    README.md           # the goal, dataset, metric, and protocol
    task.py             # the task definition (data split, metric, QA floor)
    submissions/        # one .py per entry; ours are named baseline_*.py
    SCOREBOARD.md        # generated ranking for this task
  SCOREBOARD.md          # generated aggregate: best score per task
```

## Running it

```bash
python bench/harness.py                    # run every task, refresh all boards
python bench/harness.py digits_classification   # run a single task
```

Each submission runs in its own subprocess with a timeout, so a crashing,
hanging, or rule-breaking entry cannot take down the run or affect anyone else's
score. The harness statically checks every submission's imports first and marks
any that reach outside `nupyml` / `numpy` / `scipy` / the stdlib as an
`import_violation` — they are not scored.

## The tasks

| Task | Kind | Metric | What it exercises |
|------|------|--------|-------------------|
| `digits_classification` | supervised | accuracy | multiclass classifiers |
| `moons_classification` | supervised | accuracy | nonlinear boundaries |
| `diabetes_regression` | supervised | R² | regressors (hard, low ceiling) |
| `fraud_imbalanced` | supervised | macro-F1 | rare-class handling |
| `blobs_clustering` | clustering | adjusted Rand index | unsupervised clustering |
| `sine_forecast` | forecast | RMSE (lower is better) | time-series forecasting |

## How scoring works

Each `task.py` declares its `KIND`, `METRIC`, whether higher is better, and a
`MIN_SCORE` — a QA floor every baseline must clear. The harness runs each
submission, scores its prediction against a held-out test set the submission
never sees, and ranks the entries. The `MIN_SCORE` is a sanity floor for catching
regressions, not the target — see each task's README for what a *strong* score
looks like.

## Add your own

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: drop a single `.py` exposing a
`solve(...)` function into a task's `submissions/`, run the harness, and open a
PR. New tasks are welcome too.
