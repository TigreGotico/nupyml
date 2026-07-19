"""Make hierarchical forecasts COHERENT -- the parts sum to the whole"""
import numpy as np


def reconcile_forecasts(base_forecasts, S, method="mint"):
    """Make hierarchical forecasts COHERENT -- the parts sum to the whole
    (Hyndman/Wickramasuriya).

    Forecast a total, its regions, and their stores separately and they will not
    add up. Reconciliation projects the independent ("base") forecasts onto the set
    that respects the hierarchy's summing constraints, encoded by the summing matrix
    ``S`` (each row = one node as a sum of the bottom-level series).

    * ``bottom_up`` -- trust only the leaves and sum them up.
    * ``ols`` -- least-squares projection onto the coherent subspace.
    * ``mint`` -- the minimum-trace optimal combination: the reconciliation with the
      smallest forecast-error variance (here with a diagonal error covariance).

    ``base_forecasts`` is ``(n_nodes,)`` or ``(n_nodes, horizon)``; returns coherent
    forecasts of the same shape.
    """
    S = np.asarray(S, float)
    yhat = np.atleast_2d(np.asarray(base_forecasts, float))
    if yhat.shape[0] != S.shape[0]:
        yhat = yhat.T
    n_bottom = S.shape[1]
    if method == "bottom_up":
        bottom = yhat[-n_bottom:]                        # leaves are the last rows
        P = np.hstack([np.zeros((n_bottom, S.shape[0] - n_bottom)), np.eye(n_bottom)])
    elif method == "ols":
        P = np.linalg.inv(S.T @ S) @ S.T
    elif method == "mint":
        W = np.diag(S.sum(axis=1))                       # diagonal error scale
        Wi = np.linalg.inv(W)
        P = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
    else:
        raise ValueError(f"unknown method {method!r}")
    reconciled = S @ (P @ yhat)
    return reconciled.ravel() if reconciled.shape[1] == 1 else reconciled


__all__ = ["reconcile_forecasts"]
