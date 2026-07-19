"""Symbolic Aggregate approXimation: turn a series into a short STRING"""
import numpy as np


def sax(y, n_segments=8, alphabet_size=4):
    """Symbolic Aggregate approXimation: turn a series into a short STRING
    (Lin et al., 2003).

    SAX compresses a series to symbols in two steps: PAA averages it into
    ``n_segments`` equal pieces (dimensionality reduction), then each piece's value
    is mapped to a letter by Gaussian BREAKPOINTS that split a standard normal into
    equal-probability bins (after z-normalising the series). The result is a tiny
    string that supports fast indexing, motif discovery, and a distance that lower-
    bounds the true one -- the representation behind much of symbolic time-series
    mining. Returns the symbol list.
    """
    from scipy.stats import norm
    y = np.asarray(y, float)
    y = (y - y.mean()) / (y.std() + 1e-12)             # z-normalise
    seg = np.array_split(y, n_segments)
    paa = np.array([s.mean() for s in seg])            # piecewise aggregate approx
    breaks = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    idx = np.searchsorted(breaks, paa)
    return [chr(ord("a") + i) for i in idx]


__all__ = ["sax"]
