"""Subprocess entry point: run ONE submission in isolation.

Invoked by the harness as::

    python bench/_run_one.py <submission.py> <in.npz> <out.npz> <KIND>

It loads the split the parent prepared, imports the submission by path, calls its
``solve`` with the arguments for the task KIND, and saves ``y_pred`` for the
parent to score. Anything that goes wrong -- an exception, a hang, a banned
import that slipped the static check -- stays contained in this process; the
parent sees a non-zero exit or a timeout, never a corrupted run.

The held-out labels are NOT present in ``in.npz``: this process is structurally
unable to see the answers it will be graded against.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_submission(path):
    spec = importlib.util.spec_from_file_location("_bench_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise AttributeError("submission must define a solve(...) function")
    return module


def main():
    submission_path, in_npz, out_npz, kind = sys.argv[1:5]
    module = _load_submission(submission_path)
    data = np.load(in_npz, allow_pickle=True)

    if kind == "supervised":
        y_pred = module.solve(data["X_train"], data["y_train"], data["X_test"])
    elif kind == "clustering":
        y_pred = module.solve(data["X"])
    elif kind == "forecast":
        y_pred = module.solve(data["y_history"], int(data["horizon"]))
    elif kind == "ranking":
        y_pred = module.solve(data["X_train"], data["y_train"],
                              data["groups_train"], data["X_test"],
                              data["groups_test"])
    elif kind == "survival":
        y_pred = module.solve(data["X_train"], data["durations_train"],
                              data["events_train"], data["X_test"])
    elif kind == "density":
        y_pred = module.solve(data["X_train"], data["X_test"])
    else:
        raise ValueError(f"unknown KIND {kind!r}")

    np.savez(out_npz, y_pred=np.asarray(y_pred))


if __name__ == "__main__":
    main()
