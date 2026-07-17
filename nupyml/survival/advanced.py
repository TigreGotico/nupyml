"""Survival analysis beyond Cox: metrics, random survival forests, AFT models.

The core module has Kaplan-Meier, Nelson-Aalen and Cox. These add the standard
evaluation metric (concordance) and two models that drop Cox's assumptions --
random survival forests (no proportional-hazards, no linearity) and accelerated
failure time (a different, sometimes more natural, way to model time).
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_X_y, check_array, check_random_state


def concordance_index(durations, events, risk_scores):
    """Harrell's C-index: does the model rank who-fails-first correctly?

    WHY NOT JUST ACCURACY OR MSE
    ----------------------------
    Survival predictions cannot be scored like regressions -- censored subjects
    have no observed time to compare against, so squared error is undefined for
    them. The C-index sidesteps this by asking a RANKING question over COMPARABLE
    pairs: of two subjects, if one is known to have failed before the other, did
    the model give that one the higher risk? It counts the fraction of such
    orderable pairs the model gets right.

    Censoring is handled by only comparing pairs whose order is actually known (a
    subject censored at time 5 can be compared to one who FAILED at time 3, but
    not to one censored at time 8). C = 1 is perfect ranking, 0.5 is random --
    the survival analogue of ROC-AUC, and the standard way survival models are
    reported.
    """
    durations = np.asarray(durations, float)
    events = np.asarray(events, int)
    risk = np.asarray(risk_scores, float)
    n = len(durations)
    concordant = permissible = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # a pair is comparable only if i is known to fail before j
            if events[i] == 1 and durations[i] < durations[j]:
                permissible += 1
                # higher risk should go to the one who failed first (i)
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


class RandomSurvivalForest(BaseEstimator):
    """A random forest for censored time-to-event data.

    THE POINT: DROP COX'S ASSUMPTIONS
    ---------------------------------
    Cox regression assumes proportional hazards (every covariate multiplies risk
    by a time-CONSTANT factor) and a linear log-hazard. Both are often false. A
    random survival forest inherits the forest's freedom -- non-linear, automatic
    interactions, no proportional-hazards assumption -- and adapts it to
    censoring by splitting on the LOG-RANK statistic (the split that most
    separates the two children's survival curves) instead of Gini or variance,
    and predicting each leaf's cumulative hazard (Nelson-Aalen) rather than a
    class or a mean.

    So it is "a random forest, but the split criterion and the leaf estimate
    understand censored data". Predicts a RISK SCORE per subject (the total
    cumulative hazard), rankable by the C-index.

    Ishwaran et al. (2008). This is a compact version using the log-rank split.
    """

    def __init__(self, n_estimators=50, max_depth=4, min_samples_leaf=10,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _logrank(self, dur, ev, mask):
        """Log-rank statistic between the two groups a candidate split makes.

        At each event time it compares observed vs expected failures in one group
        under the null of equal survival; the summed, variance-scaled discrepancy
        is large when the two groups' survival curves diverge -- exactly the split
        that best separates risk.
        """
        left, right = mask, ~mask
        if left.sum() < 1 or right.sum() < 1:
            return 0.0
        times = np.unique(dur[ev == 1])
        obs_left = exp_left = var = 0.0
        for t in times:
            at_risk = dur >= t
            n_risk = at_risk.sum()
            if n_risk < 2:
                continue
            d = ((dur == t) & (ev == 1)).sum()
            n_risk_left = (at_risk & left).sum()
            obs_left += ((dur == t) & (ev == 1) & left).sum()
            exp_left += d * n_risk_left / n_risk
            var += (d * (n_risk_left / n_risk) * (1 - n_risk_left / n_risk)
                    * (n_risk - d) / (n_risk - 1)) if n_risk > 1 else 0
        return (obs_left - exp_left) ** 2 / var if var > 0 else 0.0

    def _grow(self, X, dur, ev, depth, rng):
        node = {"chf": self._cum_hazard(dur, ev)}
        if depth >= self.max_depth or len(dur) < 2 * self.min_samples_leaf:
            return node
        n_feat = X.shape[1]
        k = int(np.sqrt(n_feat)) if self.max_features == "sqrt" else n_feat
        features = rng.choice(n_feat, size=min(k, n_feat), replace=False)
        best = (0.0, None, None)
        for f in features:
            for thr in np.percentile(X[:, f], [25, 50, 75]):
                mask = X[:, f] <= thr
                if mask.sum() < self.min_samples_leaf or \
                        (~mask).sum() < self.min_samples_leaf:
                    continue
                score = self._logrank(dur, ev, mask)      # best-separating split
                if score > best[0]:
                    best = (score, f, thr)
        if best[1] is None:
            return node
        _, f, thr = best
        mask = X[:, f] <= thr
        node.update(feature=f, threshold=thr,
                    left=self._grow(X[mask], dur[mask], ev[mask], depth + 1, rng),
                    right=self._grow(X[~mask], dur[~mask], ev[~mask], depth + 1, rng))
        return node

    def _cum_hazard(self, dur, ev):
        """Total Nelson-Aalen cumulative hazard for a leaf's subjects."""
        times = np.unique(dur[ev == 1])
        H = 0.0
        for t in times:
            at_risk = (dur >= t).sum()
            if at_risk > 0:
                H += ((dur == t) & (ev == 1)).sum() / at_risk
        return H

    def fit(self, X, durations, events):
        X = check_array(X)
        dur = np.asarray(durations, float)
        ev = np.asarray(events, int)
        rng = check_random_state(self.random_state)
        self.trees_ = []
        n = len(X)
        for _ in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True)        # bootstrap sample
            self.trees_.append(self._grow(X[idx], dur[idx], ev[idx], 0, rng))
        return self

    def _predict_tree(self, node, x):
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] \
                else node["right"]
        return node["chf"]

    def predict_risk(self, X):
        """Ensemble-averaged cumulative hazard -- a rankable risk score."""
        X = check_array(X)
        return np.array([np.mean([self._predict_tree(t, x) for t in self.trees_])
                         for x in X])


class AFT(BaseEstimator):
    """Accelerated Failure Time: model log-TIME directly, not the hazard.

    THE DIFFERENT FRAME FROM COX
    ----------------------------
    Cox models the HAZARD (instantaneous risk) and its covariate MULTIPLIERS. AFT
    instead models log survival time as a linear function of the covariates plus
    noise::

        log(T) = X beta + sigma * error

    So a covariate does not scale your risk -- it ACCELERATES or DECELERATES your
    clock, stretching or compressing survival time by ``exp(beta)``. That is often
    the more intuitive story ("this treatment doubles expected survival" rather
    than "halves the hazard"), and crucially it does NOT assume proportional
    hazards, so it fits cases where Cox's assumption fails.

    Censoring enters through the likelihood: an observed failure contributes its
    density, a censored subject contributes its SURVIVAL probability (all we know
    is T exceeded the censoring time). Fitted here with a log-normal time
    distribution by maximum likelihood.
    """

    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def fit(self, X, durations, events):
        from scipy.optimize import minimize
        from scipy import stats
        X = check_array(X)
        dur = np.asarray(durations, float)
        ev = np.asarray(events, int)
        Xd = np.column_stack([np.ones(len(X)), X])
        log_t = np.log(np.clip(dur, 1e-8, None))

        def neg_ll(params):
            beta, log_sigma = params[:-1], params[-1]
            sigma = np.exp(log_sigma)
            mu = Xd @ beta
            z = (log_t - mu) / sigma
            # observed failures contribute the log-normal density; censored
            # subjects contribute log P(T > t) = log survival
            ll_event = stats.norm.logpdf(z) - log_sigma
            ll_cens = stats.norm.logsf(z)
            return -np.sum(np.where(ev == 1, ll_event, ll_cens))

        p0 = np.concatenate([np.zeros(Xd.shape[1]), [0.0]])
        p0[0] = log_t.mean()
        res = minimize(neg_ll, p0, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x[1:-1]
        self.intercept_ = res.x[0]
        self.sigma_ = np.exp(res.x[-1])
        self.params_ = res.x
        return self

    def predict_median_survival(self, X):
        """Median survival time per subject: exp(X beta), the log-normal median."""
        X = check_array(X)
        return np.exp(X @ self.coef_ + self.intercept_)

    def predict_risk(self, X):
        """A risk score = negative predicted log-time (higher risk = sooner)."""
        X = check_array(X)
        return -(X @ self.coef_ + self.intercept_)


__all__ = ["concordance_index", "RandomSurvivalForest", "AFT"]
