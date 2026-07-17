"""The nupyml benchmark harness: run submissions, score them, rank the board.

WHAT THIS IS
------------
A benchmark suite that doubles as a QA harness. Each task is a folder with a
``task.py`` (goal, metric, deterministic data split) and a ``submissions/``
folder of single-file entries. A submission is a script exposing one function --
``solve(...)`` -- and NOTHING may be imported except nupyml, numpy, scipy and the
standard library. We ship baselines; the community adds entries; the harness
ranks everyone into a scoreboard.

WHY SUBPROCESS ISOLATION
------------------------
Submissions are untrusted community code AND the QA gate. Running each one in a
child process with a timeout means a crash, an infinite loop, or a banned import
in one entry cannot take down the run or corrupt another's result. The parent
holds the held-out ``y_test`` and never hands it to the child, so a submission
cannot peek at the answers.

USAGE
-----
    python bench/harness.py                     # run every task, refresh boards
    python bench/harness.py digits_classification   # one task
"""
import ast
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent

# only these top-level modules may be imported by a submission. numpy and scipy
# are nupyml's own runtime deps, so glue code may use them; everything else --
# sklearn, xgboost, torch, pandas -- is banned. The point of the bench is that
# you must build with nupyml, not wrap another library.
_ALLOWED_THIRD_PARTY = {"nupyml", "numpy", "np", "scipy"}
_STDLIB = set(sys.stdlib_module_names)
ALLOWED_IMPORTS = _ALLOWED_THIRD_PARTY | _STDLIB

TIMEOUT_SECONDS = 300


def discover_tasks():
    """Every subdirectory of bench/ that has a ``task.py``."""
    return sorted(p for p in BENCH_DIR.iterdir()
                  if p.is_dir() and (p / "task.py").exists())


def load_task(task_dir):
    """Import a task's ``task.py`` as a module (fresh each call)."""
    spec = importlib.util.spec_from_file_location(
        f"_bench_task_{task_dir.name}", task_dir / "task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_imports(path):
    """Return the list of DISALLOWED top-level imports in a submission.

    The nupyml-only gate. Parses the source with ``ast`` (never executes it) and
    collects the root package of every import; anything outside the allowlist is
    a violation. An empty list means the submission is clean.
    """
    tree = ast.parse(Path(path).read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # a relative import (level > 0) has no external module to police
            names = [] if node.level else [(node.module or "").split(".")[0]]
        else:
            continue
        for name in names:
            if name and name not in ALLOWED_IMPORTS:
                violations.append(name)
    return sorted(set(violations))


def _iter_submissions(task_dir):
    sub_dir = task_dir / "submissions"
    if not sub_dir.exists():
        return []
    return sorted(p for p in sub_dir.glob("*.py") if not p.name.startswith("_"))


def run_submission(task_dir, submission_path, timeout=TIMEOUT_SECONDS):
    """Run one submission in a child process; return (score, runtime, status).

    status is one of: ok, import_violation, error, timeout, invalid_output.
    """
    task = load_task(task_dir)

    # the import gate first -- a banned import never gets to run
    bad = check_imports(submission_path)
    if bad:
        return None, 0.0, f"import_violation({','.join(bad)})"

    # write the child's inputs to a temp npz. The held-out y_test stays HERE,
    # in the parent, so the submission cannot see the answers it is graded on.
    data = task.load()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_npz, out_npz = tmp / "in.npz", tmp / "out.npz"
        kind = task.KIND
        if kind == "supervised":
            X_train, y_train, X_test, y_test = data
            np.savez(in_npz, X_train=X_train, y_train=y_train, X_test=X_test)
        elif kind == "clustering":
            X, y_test = data
            np.savez(in_npz, X=X)
        elif kind == "forecast":
            y_history, y_test = data
            np.savez(in_npz, y_history=y_history, horizon=len(y_test))
        else:
            return None, 0.0, f"error(unknown KIND {kind!r})"

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(BENCH_DIR / "_run_one.py"),
                 str(submission_path), str(in_npz), str(out_npz), kind],
                capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        except subprocess.TimeoutExpired:
            return None, timeout, "timeout"
        runtime = time.perf_counter() - start

        if proc.returncode != 0 or not out_npz.exists():
            msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no output"
            return None, runtime, f"error({msg[:80]})"

        y_pred = np.load(out_npz)["y_pred"]
        if len(y_pred) != len(y_test):
            return None, runtime, "invalid_output(length mismatch)"

    try:
        score = float(task.metric(y_test, y_pred))
    except Exception as exc:  # a metric that chokes on the prediction shape/type
        return None, runtime, f"invalid_output({str(exc)[:60]})"
    return score, runtime, "ok"


def score_task(task_dir, timeout=TIMEOUT_SECONDS, verbose=True):
    """Run every submission for a task and write its SCOREBOARD.md."""
    task = load_task(task_dir)
    results = []
    for sub in _iter_submissions(task_dir):
        score, runtime, status = run_submission(task_dir, sub, timeout)
        results.append({"name": sub.stem, "score": score,
                        "runtime": runtime, "status": status})
        if verbose:
            s = f"{score:.4f}" if score is not None else "  --  "
            print(f"  {task_dir.name:24} {sub.stem:26} {s}  {status}")

    # rank the ones that produced a score, in the task's preferred direction
    ranked = sorted((r for r in results if r["score"] is not None),
                    key=lambda r: r["score"], reverse=task.HIGHER_IS_BETTER)
    failed = [r for r in results if r["score"] is None]
    _write_task_board(task_dir, task, ranked, failed)
    return results


def _write_task_board(task_dir, task, ranked, failed):
    arrow = "higher is better" if task.HIGHER_IS_BETTER else "lower is better"
    lines = [f"# Scoreboard — {task_dir.name}", "",
             f"**Goal:** {task.GOAL}", "",
             f"**Metric:** `{task.METRIC}` ({arrow}) · **QA floor:** "
             f"`{task.MIN_SCORE}`", "",
             "| Rank | Submission | Score | Runtime (s) |",
             "|-----:|------------|------:|------------:|"]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['score']:.4f} | {r['runtime']:.2f} |")
    if failed:
        lines += ["", "### Did not score", "",
                  "| Submission | Status |", "|------------|--------|"]
        for r in failed:
            lines.append(f"| `{r['name']}` | {r['status']} |")
    lines += ["", "_Regenerate with `python bench/harness.py "
              f"{task_dir.name}`._", ""]
    (task_dir / "SCOREBOARD.md").write_text("\n".join(lines))


def _write_aggregate(all_results):
    lines = ["# nupyml benchmark scoreboard", "",
             "Best score per task. Regenerate with `python bench/harness.py`.",
             "", "| Task | Metric | Best submission | Best score |",
             "|------|--------|-----------------|-----------:|"]
    for task_dir, task, results in all_results:
        scored = [r for r in results if r["score"] is not None]
        if scored:
            best = (max if task.HIGHER_IS_BETTER else min)(
                scored, key=lambda r: r["score"])
            lines.append(f"| [{task_dir.name}]({task_dir.name}/SCOREBOARD.md) "
                         f"| `{task.METRIC}` | `{best['name']}` "
                         f"| {best['score']:.4f} |")
        else:
            lines.append(f"| [{task_dir.name}]({task_dir.name}/SCOREBOARD.md) "
                         f"| `{task.METRIC}` | — | — |")
    lines.append("")
    (BENCH_DIR / "SCOREBOARD.md").write_text("\n".join(lines))


def main(argv):
    tasks = discover_tasks()
    if argv:
        wanted = set(argv)
        tasks = [t for t in tasks if t.name in wanted]
        if not tasks:
            print(f"no such task(s): {', '.join(argv)}")
            return 1
    all_results = []
    for task_dir in tasks:
        print(f"[{task_dir.name}]")
        results = score_task(task_dir)
        all_results.append((task_dir, load_task(task_dir), results))
    _write_aggregate(all_results)
    print(f"\nWrote {len(all_results)} scoreboard(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
