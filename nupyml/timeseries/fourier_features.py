"""Sine/cosine SEASONAL features for a given period (a Fourier basis)."""
import numpy as np


def fourier_features(length, period, n_harmonics=3):
    """Sine/cosine SEASONAL features for a given period (a Fourier basis).

    Model smooth seasonality by regressing on a few Fourier harmonics of the
    seasonal period instead of one dummy per season: for period ``p`` and harmonic
    ``h``, the pair ``sin(2*pi*h*t/p), cos(2*pi*h*t/p)``. A handful of harmonics
    captures a smooth yearly/weekly cycle with far fewer parameters than seasonal
    dummies, and it extrapolates. Returns a ``(length, 2*n_harmonics)`` design
    matrix to feed any regressor.
    """
    t = np.arange(length)
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * t / period))
        cols.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(cols)


__all__ = ["fourier_features"]
