"""The benchmark suite doubles as end-to-end QA.

Every baseline in bench/ is run through the real harness (subprocess isolation
and all) and must clear its task's MIN_SCORE floor. Because the baselines
exercise dozens of estimators on real tasks, a change that quietly breaks
accuracy fails here as a scoreboard drop -- which is the whole point of building
the bench as QA, not just as a leaderboard.
"""
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH_DIR))

import harness  # noqa: E402


def _baseline_cases():
    """(task_dir, submission_path) for every baseline_*.py in the suite."""
    cases = []
    for task_dir in harness.discover_tasks():
        for sub in sorted((task_dir / "submissions").glob("baseline_*.py")):
            cases.append((task_dir, sub))
    return cases


BASELINES = _baseline_cases()


def test_the_suite_has_tasks_and_baselines():
    """Guard against the discovery silently finding nothing."""
    assert len(harness.discover_tasks()) >= 6
    assert len(BASELINES) >= 20


@pytest.mark.parametrize("task_dir,submission",
                         BASELINES, ids=[f"{t.name}/{s.stem}" for t, s in BASELINES])
def test_baseline_clears_its_floor(task_dir, submission):
    """Each baseline runs cleanly and beats its task's QA floor."""
    task = harness.load_task(task_dir)
    score, runtime, status = harness.run_submission(task_dir, submission,
                                                    timeout=180)
    assert status == "ok", f"{submission.name} did not run: {status}"
    if task.HIGHER_IS_BETTER:
        assert score >= task.MIN_SCORE, \
            f"{submission.name} scored {score:.4f} < floor {task.MIN_SCORE}"
    else:
        assert score <= task.MIN_SCORE, \
            f"{submission.name} scored {score:.4f} > ceiling {task.MIN_SCORE}"


# --- the nupyml-only import gate ------------------------------------------

def test_import_gate_rejects_a_banned_library(tmp_path):
    """A submission importing sklearn must be flagged, not scored."""
    bad = tmp_path / "cheater.py"
    bad.write_text("import sklearn\n"
                   "def solve(X_train, y_train, X_test):\n"
                   "    return y_train[:len(X_test)]\n")
    assert "sklearn" in harness.check_imports(bad)


def test_import_gate_rejects_from_import(tmp_path):
    bad = tmp_path / "cheater2.py"
    bad.write_text("from xgboost import XGBClassifier\n"
                   "def solve(a, b, c):\n    return c\n")
    assert "xgboost" in harness.check_imports(bad)


def test_import_gate_accepts_a_clean_submission(tmp_path):
    """nupyml + numpy + scipy + stdlib are all allowed."""
    good = tmp_path / "clean.py"
    good.write_text("import numpy as np\n"
                    "import itertools\n"
                    "from nupyml.svm import SVC\n"
                    "from scipy.linalg import svd\n"
                    "def solve(X_train, y_train, X_test):\n"
                    "    return SVC().fit(X_train, y_train).predict(X_test)\n")
    assert harness.check_imports(good) == []


def test_import_violation_is_not_scored(tmp_path):
    """End to end: a banned entry gets status import_violation and no score."""
    task_dir = harness.discover_tasks()[0]
    bad = task_dir / "submissions" / "_qa_cheater.py"
    bad.write_text("import sklearn\ndef solve(a, b, c):\n    return c[:, 0]\n")
    try:
        score, _, status = harness.run_submission(task_dir, bad, timeout=30)
        assert score is None
        assert status.startswith("import_violation")
    finally:
        bad.unlink()


# --- the harness handles broken submissions gracefully --------------------

def test_a_crashing_submission_is_isolated(tmp_path):
    """A submission that raises must be reported as error, not crash the run."""
    task_dir = next(t for t in harness.discover_tasks()
                    if harness.load_task(t).KIND == "supervised")
    boom = task_dir / "submissions" / "_qa_boom.py"
    boom.write_text("def solve(X_train, y_train, X_test):\n"
                    "    raise RuntimeError('boom')\n")
    try:
        score, _, status = harness.run_submission(task_dir, boom, timeout=30)
        assert score is None
        assert status.startswith("error")
    finally:
        boom.unlink()
