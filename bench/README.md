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
| `ts_classification` | supervised | accuracy | time-series classification (shape at random phase) |
| `multilabel_emotions` | supervised | macro-F1 | correlated multi-label prediction |
| `anomaly_detection` | clustering | ROC-AUC | unsupervised outlier ranking |
| `ratings_recommender` | supervised | RMSE (lower is better) | latent-factor recommendation |
| `text_classification` | supervised | accuracy | text pipelines (tf-idf/bag-of-words + classifier) |
| `image_feature_classification` | supervised | accuracy | hand-crafted image features (HOG/LBP) |
| `learning_to_rank` | ranking | NDCG@10 | query-grouped learning to rank |
| `survival_risk` | survival | concordance index | risk ranking under censoring |
| `matrix_completion` | supervised | RMSE (lower is better) | low-rank matrix completion |
| `semi_supervised` | supervised | accuracy | learning from few labels + unlabeled (-1 sentinel) |
| `multi_output_regression` | supervised | avg R² | vector-valued regression |
| `density_estimation` | density | mean log-likelihood | fit p(x), score held-out points |
| `symbolic_regression` | supervised | R² | recover a nonlinear formula |
| `keypoint_matching` | supervised | accuracy | shape recognition from image descriptors |
| `intermittent_forecast` | forecast | RMSE (lower is better) | sparse intermittent-demand forecasting |
| `ood_detection` | clustering | ROC-AUC | out-of-distribution scoring |
| `volatility_forecast` | forecast | RMSE (lower is better) | forecast return variance (GARCH vs static) |
| `link_prediction` | supervised | ROC-AUC | rank true edges above non-edges from graph features |
| `hierarchical_forecast` | forecast | RMSE (lower is better) | forecast a part/total hierarchy coherently |
| `image_segmentation` | clustering | adjusted Rand index | segment an image's pixels into regions |
| `regime_detection` | clustering | adjusted Rand index | segment a series into its hidden regimes |
| `timeseries_anomaly` | clustering | ROC-AUC | score anomalous timesteps in a series |
| `multivariate_forecast` | forecast | RMSE (lower is better) | jointly forecast coupled series (VAR) |
| `node_classification` | supervised | accuracy | label graph nodes from adjacency features |

Two KINDs beyond `supervised`/`clustering`/`forecast`:

- **`ranking`** — `solve(X_train, y_train, groups_train, X_test, groups_test) ->
  scores`. `groups_*` list the size of each contiguous query block; the metric
  scores per query (NDCG). Relevances (`y_test`) stay held out.
- **`survival`** — `solve(X_train, durations_train, events_train, X_test) ->
  risk_scores` (higher = fails sooner). `events=0` means censored; the held-out
  `(durations_test, events_test)` feed the concordance index.
- **`density`** — `solve(X_train, X_test) -> log_probs`, one log-density per test
  point. Unsupervised; the score is the mean held-out log-likelihood. Entries
  must return proper (normalised) log-densities.

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
