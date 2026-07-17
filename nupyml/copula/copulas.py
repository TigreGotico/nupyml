"""Copula families: fit dependence structure, sample from it.

Each copula fits to data with values already on [0,1] (or converts via empirical
ranks), exposes ``sample`` to draw new dependent uniforms, and reports the
dependence parameter. The marginals are handled separately -- that separation is
the whole point of a copula.
"""
import numpy as np
from scipy import stats

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


def _to_uniform(X):
    """Convert each column to uniform [0,1] via its empirical rank (the 'pseudo-
    observations' a copula is fitted on -- this strips out the marginals)."""
    X = check_array(X)
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    return ranks / (len(X) + 1)


class GaussianCopula(BaseEstimator):
    """Dependence from a correlation matrix -- but NO tail dependence.

    Map the uniform inputs through the inverse normal CDF, fit a correlation
    matrix on the result, and that correlation IS the dependence structure. It is
    the simplest copula and the natural default -- but its defining limitation is
    that its tails are INDEPENDENT: no matter how high the correlation, the
    probability that both variables hit their extremes together vanishes. For risk
    modelling that is exactly the wrong assumption (extremes are when things crash
    together), which is the ``StudentTCopula``/Archimedean case for using
    something else.
    """

    def fit(self, X):
        U = _to_uniform(X)
        # to the normal scale, where the dependence is a plain correlation
        Z = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
        self.correlation_ = np.corrcoef(Z, rowvar=False)
        self.dim_ = X.shape[1]
        return self

    def sample(self, n, random_state=None):
        rng = check_random_state(random_state)
        L = np.linalg.cholesky(self.correlation_ + 1e-8 * np.eye(self.dim_))
        Z = rng.normal(size=(n, self.dim_)) @ L.T       # correlated normals
        return stats.norm.cdf(Z)                        # back to uniforms


class StudentTCopula(BaseEstimator):
    """Like Gaussian, but with SYMMETRIC tail dependence via heavy t-tails.

    Same correlation-based structure, but the variables are coupled through a
    multivariate Student-t instead of a normal. The t's heavy tails give TAIL
    DEPENDENCE: extreme values in both variables tend to occur together, in BOTH
    directions, and how strongly is set by the degrees of freedom ``df`` (small
    df = fat tails = strong co-extremes; large df -> the Gaussian copula). This is
    the standard fix when the Gaussian copula's independent tails understate joint
    risk.
    """

    def __init__(self, df=4):
        self.df = df

    def fit(self, X):
        U = _to_uniform(X)
        Z = stats.t.ppf(np.clip(U, 1e-6, 1 - 1e-6), df=self.df)
        self.correlation_ = np.corrcoef(Z, rowvar=False)
        self.dim_ = X.shape[1]
        return self

    def sample(self, n, random_state=None):
        rng = check_random_state(self.random_state if hasattr(self, "random_state")
                                 else random_state)
        L = np.linalg.cholesky(self.correlation_ + 1e-8 * np.eye(self.dim_))
        Z = rng.normal(size=(n, self.dim_)) @ L.T
        # divide by a chi-square mixing variable -- what makes the tails heavy
        g = rng.chisquare(self.df, size=(n, 1)) / self.df
        T = Z / np.sqrt(g)
        return stats.t.cdf(T, df=self.df)


class _ArchimedeanCopula(BaseEstimator):
    """Base for bivariate Archimedean copulas (fit theta from Kendall's tau)."""

    def fit(self, X):
        U = _to_uniform(X)
        if U.shape[1] != 2:
            raise ValueError("Archimedean copulas here are bivariate")
        self.u_ = U
        tau, _ = stats.kendalltau(U[:, 0], U[:, 1])
        # each family has a closed-form link between Kendall's tau and its
        # dependence parameter theta -- how the single knob is estimated
        self.theta_ = self._theta_from_tau(tau)
        return self


class ClaytonCopula(_ArchimedeanCopula):
    """LOWER-tail dependence: variables crash together, boom independently.

    The Clayton copula concentrates dependence in the LOWER tail -- when one
    variable is small, the other tends to be small too, but their large values are
    nearly independent. That asymmetry is its whole point: it is the model for
    "assets that fall together in a crisis but rise on their own", which the
    symmetric Gaussian and t copulas cannot express. One parameter ``theta`` (> 0)
    sets the strength; theta -> 0 approaches independence.
    """

    def _theta_from_tau(self, tau):
        tau = np.clip(tau, 1e-3, 0.99)
        return 2 * tau / (1 - tau)              # Clayton's tau-theta relation

    def sample(self, n, random_state=None):
        rng = check_random_state(random_state)
        theta = self.theta_
        u = rng.uniform(size=n)
        w = rng.uniform(size=n)
        # conditional-sampling: draw u, then v from the conditional Clayton CDF
        v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
        return np.column_stack([u, v])


class GumbelCopula(_ArchimedeanCopula):
    """UPPER-tail dependence: variables spike together, crash independently.

    The mirror of Clayton: Gumbel concentrates dependence in the UPPER tail --
    joint extremes on the high side co-occur, while low values are nearly
    independent. The model for "things that boom together" (correlated maxima:
    floods, insurance claims, simultaneous highs). Parameter ``theta`` >= 1;
    theta = 1 is independence.
    """

    def _theta_from_tau(self, tau):
        tau = np.clip(tau, 1e-3, 0.99)
        return 1 / (1 - tau)                   # Gumbel's tau-theta relation

    def sample(self, n, random_state=None):
        rng = check_random_state(random_state)
        theta = self.theta_
        # sample via the positive-stable mixing variable (Marshall-Olkin)
        beta = (np.cos(np.pi / (2 * theta))) ** theta
        unif = rng.uniform(0, np.pi, size=n)
        expo = rng.exponential(size=n)
        s = (np.sin((1 - 1 / theta) * unif) / expo) ** (theta - 1) \
            * np.sin(unif / theta) / (np.sin(unif) ** theta)
        s = np.abs(s) + 1e-9
        e1, e2 = rng.exponential(size=n), rng.exponential(size=n)
        u = np.exp(-(e1 / s) ** (1 / theta))
        v = np.exp(-(e2 / s) ** (1 / theta))
        return np.column_stack([np.clip(u, 1e-6, 1 - 1e-6),
                                np.clip(v, 1e-6, 1 - 1e-6)])


class FrankCopula(_ArchimedeanCopula):
    """SYMMETRIC dependence with NO tail dependence.

    Frank couples the variables symmetrically (like Gaussian) but is Archimedean
    (a single parameter, ``theta``). Its distinguishing feature is that it has
    NEITHER upper nor lower tail dependence -- extremes decouple in both
    directions -- so it models data that is dependent in the middle but whose
    extremes are independent. ``theta`` can be any nonzero real; its sign gives
    positive or negative dependence, and theta -> 0 is independence.
    """

    def _theta_from_tau(self, tau):
        # Frank's tau-theta link has no closed form; a simple monotone proxy
        return np.clip(tau * 10, -35, 35) if abs(tau) > 1e-3 else 0.1

    def sample(self, n, random_state=None):
        rng = check_random_state(random_state)
        theta = self.theta_
        u = rng.uniform(size=n)
        w = rng.uniform(size=n)
        # inverse conditional Frank CDF
        v = -1 / theta * np.log(
            1 + w * (1 - np.exp(-theta)) /
            (w * (np.exp(-theta * u) - 1) - np.exp(-theta * u)))
        return np.column_stack([u, np.clip(v, 1e-6, 1 - 1e-6)])


__all__ = ["GaussianCopula", "StudentTCopula", "ClaytonCopula", "GumbelCopula",
           "FrankCopula"]
