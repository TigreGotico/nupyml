# Contributing to the nupyml benchmark

## Adding a submission

1. Pick a task folder (e.g. `bench/digits_classification/`).
2. Read its `README.md` and `task.py` — note the `KIND`, which decides the
   function signature you must expose.
3. Create `bench/<task>/submissions/<your_name>.py` with a single function:

   | Task `KIND` | Your function |
   |-------------|---------------|
   | `supervised` | `def solve(X_train, y_train, X_test): return y_pred` |
   | `clustering` | `def solve(X): return labels` |
   | `forecast`   | `def solve(y_history, horizon): return y_future` |

4. Import **only** `nupyml`, `numpy`, `scipy`, and the standard library. Anything
   else is rejected by the harness.
5. Run it:

   ```bash
   python bench/harness.py <task>
   ```

   Your entry appears on `bench/<task>/SCOREBOARD.md`.
6. Open a PR with your submission file (and the regenerated scoreboards).

### Example

```python
# bench/digits_classification/submissions/my_entry.py
from nupyml.svm import SVC

def solve(X_train, y_train, X_test):
    model = SVC(kernel="rbf", C=5.0, gamma=0.03).fit(X_train, y_train)
    return model.predict(X_test)
```

### Rules

- **nupyml-only.** The whole point is to solve the task with nupyml. Reaching for
  another ML library defeats it, and the harness will not score such an entry.
- **No peeking.** Your `solve` receives only the training data (and, for
  supervised tasks, the test *features*). The test labels stay in the harness;
  you cannot access them.
- **Be deterministic.** Set `random_state` where a model takes one, so your score
  is reproducible.
- **Stay within the timeout** (default 300 s per submission).

## Adding a task

Create `bench/<new_task>/` with:

- `task.py` declaring:
  - `KIND` — `"supervised"`, `"clustering"`, or `"forecast"`
  - `GOAL`, `METRIC` (strings), `HIGHER_IS_BETTER` (bool), `MIN_SCORE` (float)
  - `load()` — returns the split **deterministically** (fixed seed). For
    `supervised`: `(X_train, y_train, X_test, y_test)`; for `clustering`:
    `(X, y_true)`; for `forecast`: `(y_history, y_future)`. Reuse
    `nupyml.datasets` loaders / `make_*` generators so no data files are needed.
  - `metric(y_true, y_pred)` — returns a float.
- `README.md` — the goal, dataset, metric, and what a strong score looks like.
- `submissions/` with at least one `baseline_*.py` so the board starts populated.

Then `python bench/harness.py <new_task>` and commit the generated
`SCOREBOARD.md`.

The QA test (`test/test_bench.py`) will pick the new task up automatically and
assert every baseline clears its `MIN_SCORE` — so set the floor below what your
baselines actually achieve, with a little margin.
