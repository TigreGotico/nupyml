"""AutoML-lite: Hyperband and a TPE-based search.

``HalvingGridSearchCV`` already does successive halving from one fixed starting
budget. These two go further on the "how to spend a fixed search budget" problem:
Hyperband hedges over HOW aggressively to halve, and TPE learns a MODEL of which
hyperparameters have been working and samples toward them.
"""
import numpy as np

from ..base import clone
from ..utils import check_random_state
from ._search import _BaseSearchCV, _index, ParameterSampler


class HyperbandSearchCV(_BaseSearchCV):
    """Hyperband: hedge across DIFFERENT successive-halving aggressions.

    THE DILEMMA HALVING CANNOT RESOLVE ALONE
    ----------------------------------------
    Successive halving must choose: many configs each given little budget (great
    if cheap-to-evaluate rankings hold), or few configs each given lots (safer
    when early performance misleads). Pick wrong and you either starve the winner
    or barely explore. Hyperband refuses to choose: it runs SEVERAL brackets, each
    a halving race with a different (n_configs, starting_budget) trade-off, from
    "very many, very cheap" to "few, full budget". One bracket is right for any
    given problem, and running them all costs only a constant factor more than a
    single halving run -- a near-optimal hedge with no tuning.

    Resource = the number of training samples a config is allowed to see; ``eta``
    is the halving factor, ``max_resource`` the fraction (0-1) of the data at full
    budget.

    Li et al. (2017).
    """

    def __init__(self, estimator, param_distributions, max_resource=1.0, eta=3,
                 min_resource=30, cv=None, scoring=None, refit=True,
                 random_state=None):
        super().__init__(estimator, cv, scoring, refit)
        self.param_distributions = param_distributions
        self.max_resource = max_resource
        self.eta = eta
        self.min_resource = min_resource
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        max_r = max(2, int(self.max_resource * n))
        # every subsample must be large enough for the CV split to see each class
        floor = min(max_r, self.min_resource)
        s_max = int(np.log(max(2, max_r // floor)) / np.log(self.eta))
        results = {"params": [], "mean_test_score": [], "bracket": [],
                   "n_resources": []}
        best_score, best_params = -np.inf, None

        for s in range(s_max, -1, -1):
            # bracket s: n_configs configs, each starting at resource r0
            n_configs = int(np.ceil((s_max + 1) / (s + 1) * self.eta ** s))
            r0 = max(floor, int(max_r * self.eta ** (-s)))
            configs = list(ParameterSampler(self.param_distributions, n_configs,
                                            random_state=rng.randint(2 ** 31 - 1)))
            for i in range(s + 1):              # the halving rounds of this bracket
                r_i = min(int(r0 * self.eta ** i), n)
                idx = rng.choice(n, size=r_i, replace=False)
                Xs = _index(X, idx)
                ys = None if y is None else np.asarray(y)[idx]
                res, _, _ = self._evaluate(configs, Xs, ys)
                scores = np.asarray(res["mean_test_score"])
                for p, sc in zip(res["params"], scores):
                    results["params"].append(p)
                    results["mean_test_score"].append(float(sc))
                    results["bracket"].append(s)
                    results["n_resources"].append(r_i)
                    if sc > best_score:
                        best_score, best_params = float(sc), p
                keep = max(1, int(len(configs) / self.eta))
                order = np.argsort(-scores)[:keep]
                configs = [res["params"][j] for j in order]
        return self._finalize(results, best_score, best_params, X, y)


class TPESearchCV(_BaseSearchCV):
    """Tree-structured Parzen Estimator: MODEL the good region, sample toward it.

    THE IDEA
    --------
    Random search wastes trials re-exploring parts of the space that are already
    known to be bad. TPE keeps the history of (params, score) and, each round,
    splits it into the best fraction ``gamma`` (the "good" trials) and the rest.
    It fits a density ``l(x)`` to the good ones and ``g(x)`` to the bad, then
    proposes the candidate that maximises ``l(x)/g(x)`` -- the point most typical
    of good trials and least typical of bad ones. That ratio is exactly what
    maximises Expected Improvement, so TPE is Bayesian optimisation done with two
    cheap densities instead of a Gaussian process, which is why it scales to many
    dimensions and mixed parameter types.

    Here each parameter is either a list (categorical: densities are smoothed
    counts) or a ``(low, high)`` numeric range (Gaussian-kernel density). Proposals
    are drawn from the good density and ranked by the ratio.

    Bergstra et al. (2011).
    """

    def __init__(self, estimator, param_distributions, n_iter=30, n_startup=10,
                 gamma=0.25, n_candidates=24, cv=None, scoring=None, refit=True,
                 random_state=None):
        super().__init__(estimator, cv, scoring, refit)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.n_startup = n_startup
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.random_state = random_state

    def _sample_random(self, rng):
        params = {}
        for k, v in sorted(self.param_distributions.items()):
            if isinstance(v, tuple) and len(v) == 2:      # numeric (low, high)
                params[k] = rng.uniform(v[0], v[1])
            else:                                          # categorical list
                params[k] = v[rng.randint(len(v))]
        return params

    def _density_scores(self, values, candidates, spec, rng):
        """Relative density of each candidate under a set of observed values."""
        values = np.asarray(values, dtype=object)
        if isinstance(spec, tuple) and len(spec) == 2:     # numeric KDE
            vals = values.astype(float)
            span = spec[1] - spec[0]
            bw = max(span / 5.0, 1e-6) * (len(vals) ** (-1 / 5.0) + 1e-9)
            cand = np.asarray(candidates, dtype=float)
            # mean Gaussian kernel from each candidate to the observed values
            diff = (cand[:, None] - vals[None, :]) / bw
            return np.exp(-0.5 * diff ** 2).mean(axis=1) + 1e-12
        # categorical: smoothed frequency
        levels = list(spec)
        counts = {lv: 1.0 for lv in levels}                # Laplace smoothing
        for v in values:
            counts[v] = counts.get(v, 1.0) + 1.0
        total = sum(counts.values())
        return np.array([counts[c] / total for c in candidates])

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        keys = sorted(self.param_distributions)
        history, scores = [], []
        results = {"params": [], "mean_test_score": []}
        best_score, best_params = -np.inf, None

        for t in range(self.n_iter):
            if t < self.n_startup or len(history) < 4:
                cand = self._sample_random(rng)             # warm up randomly
            else:
                # split history into good (top gamma) and bad by score
                order = np.argsort(-np.asarray(scores))
                n_good = max(1, int(np.ceil(self.gamma * len(history))))
                good = [history[i] for i in order[:n_good]]
                bad = [history[i] for i in order[n_good:]] or good
                # propose candidates from the good density, rank by l/g
                proposals = [self._sample_from(good, rng) for _ in
                             range(self.n_candidates)]
                ratio = np.ones(len(proposals))
                for k in keys:
                    spec = self.param_distributions[k]
                    cvals = [p[k] for p in proposals]
                    l = self._density_scores([h[k] for h in good], cvals, spec, rng)
                    g = self._density_scores([h[k] for h in bad], cvals, spec, rng)
                    ratio *= l / g
                cand = proposals[int(np.argmax(ratio))]

            est = clone(self.estimator).set_params(**cand)
            from . import cross_val_score
            from ._scoring import check_scoring
            sc = float(cross_val_score(est, X, y, cv=self.cv,
                                       scoring=check_scoring(self.scoring)).mean())
            history.append(cand)
            scores.append(sc)
            results["params"].append(cand)
            results["mean_test_score"].append(sc)
            if sc > best_score:
                best_score, best_params = sc, cand
        results["std_test_score"] = [0.0] * len(results["params"])
        return self._finalize(results, best_score, best_params, X, y)

    def _sample_from(self, good, rng):
        """Draw a proposal by perturbing a randomly chosen good trial."""
        base = good[rng.randint(len(good))]
        params = {}
        for k, spec in sorted(self.param_distributions.items()):
            if isinstance(spec, tuple) and len(spec) == 2:
                span = spec[1] - spec[0]
                val = base[k] + rng.normal(0, span / 5.0)
                params[k] = float(np.clip(val, spec[0], spec[1]))
            else:
                params[k] = base[k] if rng.rand() < 0.7 else spec[rng.randint(
                    len(spec))]
        return params


class BOHBSearchCV(_BaseSearchCV):
    """BOHB = Hyperband's schedule + TPE's PROPOSALS (Falkner et al., 2018).

    THE BEST OF BOTH
    ----------------
    Hyperband allocates budget well but samples configurations at RANDOM -- it
    never learns which regions of the space are good. TPE learns the good region
    but has no principled budget schedule. BOHB fuses them: it runs Hyperband's
    successive-halving brackets for the budget allocation, but replaces the random
    sampling with a TPE model fit on all configurations evaluated SO FAR. Early
    rounds explore (little data, near-random); later rounds exploit the model. The
    result converges faster than either parent.

    This version keeps a running history of (config, score) across brackets,
    fits a TPE density on it, and draws each bracket's configs from that model
    (random until enough history accrues), evaluating on sample-size budgets.
    """

    def __init__(self, estimator, param_distributions, max_resource=1.0, eta=3,
                 min_resource=30, gamma=0.25, cv=None, scoring=None, refit=True,
                 random_state=None):
        super().__init__(estimator, cv, scoring, refit)
        self.param_distributions = param_distributions
        self.max_resource = max_resource
        self.eta = eta
        self.min_resource = min_resource
        self.gamma = gamma
        self.random_state = random_state

    def _sample(self, rng, history, scores):
        # explore randomly until enough history, then propose from the TPE
        # good-density (perturb a top-gamma config); the exploit half of BOHB
        if len(history) < 8:
            p = {}
            for k, v in sorted(self.param_distributions.items()):
                p[k] = (rng.uniform(*v) if isinstance(v, tuple) and len(v) == 2
                        else v[rng.randint(len(v))])
            return p
        order = np.argsort(-np.asarray(scores))
        n_good = max(1, int(np.ceil(self.gamma * len(history))))
        base = history[order[rng.randint(n_good)]]
        p = {}
        for k, v in sorted(self.param_distributions.items()):
            if isinstance(v, tuple) and len(v) == 2:
                span = v[1] - v[0]
                p[k] = float(np.clip(base[k] + rng.normal(0, span / 5.0), v[0], v[1]))
            else:
                p[k] = base[k] if rng.rand() < 0.7 else v[rng.randint(len(v))]
        return p

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        max_r = max(2, int(self.max_resource * n))
        floor = min(max_r, self.min_resource)
        s_max = int(np.log(max(2, max_r // floor)) / np.log(self.eta))
        history, scores = [], []
        results = {"params": [], "mean_test_score": []}
        best_score, best_params = -np.inf, None

        for s in range(s_max, -1, -1):
            n_configs = int(np.ceil((s_max + 1) / (s + 1) * self.eta ** s))
            configs = [self._sample(rng, history, scores) for _ in range(n_configs)]
            r0 = max(floor, int(max_r * self.eta ** (-s)))
            for i in range(s + 1):
                r_i = min(int(r0 * self.eta ** i), n)
                idx = rng.choice(n, size=r_i, replace=False)
                Xs = _index(X, idx)
                ys = None if y is None else np.asarray(y)[idx]
                res, _, _ = self._evaluate(configs, Xs, ys)
                sc = np.asarray(res["mean_test_score"])
                for p, v in zip(res["params"], sc):
                    history.append(p); scores.append(float(v))
                    results["params"].append(p)
                    results["mean_test_score"].append(float(v))
                    if v > best_score:
                        best_score, best_params = float(v), p
                keep = max(1, int(len(configs) / self.eta))
                configs = [res["params"][j] for j in np.argsort(-sc)[:keep]]
        results["std_test_score"] = [0.0] * len(results["params"])
        return self._finalize(results, best_score, best_params, X, y)


__all__ = ["HyperbandSearchCV", "TPESearchCV", "BOHBSearchCV"]
