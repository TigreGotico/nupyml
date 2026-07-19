"""Moving-block bootstrap: resample a dependent series without breaking autocorrelation."""
import numpy as np

from ..utils import check_random_state


def block_bootstrap(series, block_size=10, n_boot=100, statistic=None,
                    random_state=None):
    """Resample a dependent series WITHOUT destroying its autocorrelation
    (Kunsch, 1989).

    The ordinary bootstrap resamples observations independently, which is exactly wrong
    for a time series -- it shatters the temporal dependence you are trying to reason
    about. The moving-block bootstrap resamples contiguous BLOCKS instead, so the
    within-block correlation structure is preserved in each replicate. This gives valid
    confidence intervals for statistics of dependent data (a mean, an autocorrelation, a
    forecast error) that the naive bootstrap would get badly wrong. Returns bootstrap
    replicates (or the statistic evaluated on each if ``statistic`` is given).
    """
    y = np.asarray(series, float).ravel()
    rng = check_random_state(random_state)
    n = len(y)
    n_blocks = int(np.ceil(n / block_size))
    out = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, n_blocks)
        rep = np.concatenate([y[s:s + block_size] for s in starts])[:n]
        out.append(statistic(rep) if statistic else rep)
    return np.array(out)


__all__ = ["block_bootstrap"]
