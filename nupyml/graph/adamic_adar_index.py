"""Common neighbours weighted by 1/log(degree): RARE shared friends count more."""
import numpy as np


def _binary(A):
    return (np.asarray(A) > 0).astype(float)


# --- link prediction (each returns an n x n score matrix) -----------------


def adamic_adar_index(A):
    """Common neighbours weighted by 1/log(degree): RARE shared friends count more.

    A mutual friend who knows everyone (a hub) is weak evidence you two are
    connected; a mutual friend with few links is strong evidence. Adamic-Adar
    down-weights each shared neighbour by ``1/log(deg)``, and it is one of the
    strongest simple predictors on social graphs.
    """
    B = _binary(A)
    deg = B.sum(axis=1)
    with np.errstate(divide="ignore"):
        w = np.where(deg > 1, 1.0 / np.log(deg), 0.0)
    S = (B * w[None, :]) @ B.T
    np.fill_diagonal(S, 0)
    return S


__all__ = ["adamic_adar_index"]
