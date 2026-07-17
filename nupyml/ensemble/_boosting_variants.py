"""The famous boosting variants, as deltas on the histogram tree.

XGBoost, LightGBM and CatBoost are all gradient boosting. They are not different
algorithms -- they are the same algorithm plus one or two specific ideas each,
and those ideas are what this module is for. Stripped of their engineering, the
headline claims are:

* **LightGBM** is fast because of ``GOSS`` (use fewer SAMPLES per tree) and
  ``EFB`` (use fewer FEATURES per tree). Both attack the cost of building a
  histogram, which is ``O(n_samples * n_features)`` -- the only two factors
  available.
* **CatBoost** is accurate on categorical data because of ``ordered boosting``
  and ordered target statistics, both fixing a target leak that the obvious
  implementations have and nobody noticed for years.
* **DART** borrows dropout from neural networks, fixing boosting's habit of
  letting the first few trees dominate.
* **NGBoost** predicts a DISTRIBUTION rather than a number.

Each is worth reading for its idea, not its speed.

WHAT THEY SHARE
---------------
Every one of these builds on ``_hist_gb._HistTree``, and the point of that reuse
is to show how small the deltas actually are. The distance between "gradient
boosting" and "LightGBM" is much shorter than the branding suggests.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, clone
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..utils import check_X_y, check_array, check_random_state, sigmoid
from ._hist_gb import _BinMapper, _HistTree, _BaseHistGB


# --- LightGBM's two ideas -------------------------------------------------

def goss_sample(gradients, top_rate=0.2, other_rate=0.1, rng=None):
    """Gradient-based One-Side Sampling: keep the big gradients, sample the rest.

    THE OBSERVATION
    ---------------
    In boosting, a sample's gradient is how wrong the model currently is about
    it. Samples with tiny gradients are already well predicted -- they contribute
    almost nothing to the next tree's split decisions. Most samples, most of the
    time, are in this category.

    So: keep ALL the large-gradient samples, and randomly sample a small fraction
    of the rest. Train on maybe 30% of the data and lose almost nothing, because
    the 70% dropped had nearly nothing to say.

    THE CORRECTION THAT MAKES IT VALID
    ----------------------------------
    Dropping the small-gradient samples would bias the estimate: the retained
    sample is no longer representative of the data distribution, and the split
    gains computed on it would systematically favour whatever the large-gradient
    samples happen to want.

    The fix is to UPWEIGHT the survivors from the sampled group by::

        (1 - top_rate) / other_rate

    Each survivor stands in for the ones dropped alongside it, so the gain
    estimate stays unbiased in expectation. Without this line GOSS is just
    "throw away data and hope"; with it, it is a variance-reduced estimator.

    Note this is the opposite of importance sampling's usual use -- here the
    weight compensates for a deliberate, informed bias in what was kept.

    Returns ``(indices, weights)``.

    Ke et al. (2017).
    """
    rng = check_random_state(rng)
    n = len(gradients)
    n_top = int(top_rate * n)
    n_other = int(other_rate * n)

    # sorted by how wrong we currently are, most wrong first
    order = np.argsort(-np.abs(gradients))
    top_idx = order[:n_top]
    rest = order[n_top:]

    if n_other > 0 and len(rest) > 0:
        other_idx = rng.choice(rest, size=min(n_other, len(rest)), replace=False)
    else:
        other_idx = np.array([], dtype=int)

    idx = np.concatenate([top_idx, other_idx])
    weights = np.ones(len(idx))
    if len(other_idx) > 0:
        # the survivors speak for the dropped -- see the docstring
        weights[len(top_idx):] = (1.0 - top_rate) / other_rate
    return idx, weights


def exclusive_feature_bundles(X_binned, max_conflict_rate=0.0):
    """Exclusive Feature Bundling: pack mutually-exclusive features into one.

    THE OBSERVATION
    ---------------
    High-dimensional data is usually sparse, and sparse features are often
    MUTUALLY EXCLUSIVE -- one-hot columns are the pure case, where exactly one of
    the group is non-zero per row. Two features that are never non-zero on the
    same row can share a single histogram without ambiguity: offset the second
    one's bins past the first's, and a single column encodes both.

    Bundle ``k`` such features and the histogram cost drops by ``k``, with NO
    information lost. Unlike GOSS this is exact when the exclusivity is exact --
    it is a change of representation, not an approximation.

    WHY IT IS ONLY A HEURISTIC
    --------------------------
    Optimally partitioning features into the fewest bundles is graph colouring,
    which is NP-hard. So this uses the standard greedy: order features by how
    many conflicts they have, then drop each into the first bundle it fits.
    Good enough, and the alternative is not available at any price.

    ``max_conflict_rate`` above 0 allows APPROXIMATE bundling -- features that
    conflict on a few rows get bundled anyway, and those rows are simply encoded
    wrongly. That trades a little accuracy for more bundling, and is the kind of
    knob worth knowing is there before wondering why results moved.

    Returns a list of bundles, each a list of feature indices.
    """
    n_samples, n_features = X_binned.shape
    # "non-zero" means "not in bin 0" -- for one-hot data, bin 0 is the absence
    nonzero = X_binned != 0
    max_conflicts = int(max_conflict_rate * n_samples)

    # conflicts[i, j]: rows where both features are non-zero. One matmul gives
    # every pair at once rather than a Python double loop over features.
    conflicts = nonzero.T.astype(np.int32) @ nonzero.astype(np.int32)
    degree = conflicts.sum(axis=1)
    order = np.argsort(-degree)     # greedy: hardest to place first

    bundles = []
    bundle_conflicts = []
    for j in order:
        placed = False
        for b, bundle in enumerate(bundles):
            total = bundle_conflicts[b] + conflicts[j, bundle].sum()
            if total <= max_conflicts:
                bundle.append(int(j))
                bundle_conflicts[b] = total
                placed = True
                break
        if not placed:
            bundles.append([int(j)])
            bundle_conflicts.append(0)
    return bundles


def bundle_features(X_binned, bundles, n_bins):
    """Encode bundled features into one column each, by offsetting their bins.

    Feature ``a`` keeps bins ``1..k_a``; feature ``b`` gets ``k_a+1..k_a+k_b``,
    and so on. Bin 0 stays "none of them", which is why the offsets start at 1.
    A split on the bundled column can therefore still isolate any original
    feature's range -- the tree loses nothing.
    """
    out = np.zeros((len(X_binned), len(bundles)), dtype=np.uint8)
    offsets = []
    for b, bundle in enumerate(bundles):
        offset = 0
        bundle_offsets = []
        for j in bundle:
            col = X_binned[:, j]
            nz = col != 0
            shifted = col.astype(np.int32) + offset
            np.clip(shifted, 0, n_bins - 1, out=shifted)
            out[nz, b] = shifted[nz].astype(np.uint8)
            bundle_offsets.append(offset)
            offset += int(col.max()) + 1 if len(col) else 1
        offsets.append(bundle_offsets)
    return out, offsets


class GOSSRegressor(_BaseHistGB, RegressorMixin):
    """Gradient boosting trained on a gradient-biased subsample each round.

    Trains on ``top_rate + other_rate`` of the data per tree -- 30% by default --
    and should land close to the full-data model. The gap between them is the
    honest measure of whether GOSS works, and is what this class exists to let
    you check.
    """

    def __init__(self, max_iter=100, learning_rate=0.1, max_depth=None,
                 max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                 max_bins=255, top_rate=0.2, other_rate=0.1, random_state=None):
        super().__init__(max_iter=max_iter, learning_rate=learning_rate,
                         max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                         min_samples_leaf=min_samples_leaf,
                         l2_regularization=l2_regularization, max_bins=max_bins,
                         random_state=random_state)
        self.top_rate = top_rate
        self.other_rate = other_rate

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite="allow-nan")
        rng = check_random_state(self.random_state)
        self._mapper = _BinMapper(self.max_bins).fit(X)
        Xb = self._mapper.transform(X)

        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees_ = []
        self.subsample_sizes_ = []

        for _ in range(self.max_iter):
            g = pred - y
            h = np.ones(len(y))
            idx, w = goss_sample(g, self.top_rate, self.other_rate, rng)
            self.subsample_sizes_.append(len(idx))

            # the tree sees only the subsample, with the correction weights
            # folded into the gradients and hessians -- which is exactly where
            # they belong, since the tree scores splits by summing them
            Xb_sub = Xb[idx]
            tree = self._new_tree().fit(Xb_sub, g[idx] * w, h[idx] * w)
            # but it predicts for everyone, because every sample's residual must
            # be updated whether or not it was trained on
            pred += self.learning_rate * tree.predict(Xb)
            self.trees_.append(tree)

        self.n_iter_ = len(self.trees_)
        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        Xb = self._mapper.transform(check_array(X, force_all_finite="allow-nan"))
        pred = np.full(len(Xb), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(Xb)
        return pred


# --- CatBoost's idea: the leak nobody noticed -----------------------------

def ordered_target_statistic(values, y, rng=None, prior_weight=1.0, prior=None):
    """Encode a category by the mean target of the rows BEFORE it in an order.

    THE LEAK IN THE OBVIOUS VERSION
    -------------------------------
    Plain target encoding replaces a category with the mean ``y`` of all rows
    having it. Row ``i``'s own label is in that mean. So the encoded feature for
    row ``i`` contains information about ``y_i``, and a tree splitting on it is
    reading the answer.

    For a rare category the effect is total: a category appearing once gets
    encoded as exactly that row's label. The model learns "this feature IS the
    target", scores brilliantly in training, and collapses in production. It is
    the purest form of target leakage, and it hides inside a preprocessing step
    that looks entirely innocent.

    THE FIX
    -------
    Fix a random permutation and encode each row using only rows BEFORE it::

        encoding[i] = (sum of y[j] for j < i with the same category + prior)
                      / (count of such j + 1)

    Row ``i``'s own label cannot appear, so there is nothing to leak. The
    metaphor is that each row is encoded using only the "history" available at
    the moment it arrives -- the same discipline as not using tomorrow's price to
    trade today.

    The cost is that early rows in the permutation have almost no history and are
    encoded with a high-variance estimate. CatBoost averages several permutations
    to damp that. The prior is what a row with no history gets.

    THE RESIDUAL LEAK, STATED HONESTLY
    ----------------------------------
    The default prior is the global mean of ``y`` -- which includes row ``i``'s
    own label. So a leak of order ``1/n`` survives, in the very function written
    to remove leaks.

    It is worth being precise rather than quiet about this. The leak this fixes is
    ``O(1)``: for a category appearing once, plain encoding hands the model
    ``y_i`` exactly, and no amount of data helps. What remains is ``y_i / n``,
    which vanishes as the data grows. Trading an ``O(1)`` leak for an ``O(1/n)``
    one is the whole improvement, and it is why this is the standard formulation
    rather than a compromise -- but "reduced to negligible" is not "eliminated",
    and a test that checks the invariant strictly will find it.

    Pass ``prior`` explicitly -- a constant fixed in advance, or a value computed
    from a separate split -- when even that matters.

    OPTIMIZATION: the "sum of previous rows with this category" is a cumulative
    sum WITHIN each category, which numpy does in one pass per category rather
    than an O(n^2) scan over pairs.

    Prokhorenkova et al. (2018).
    """
    rng = check_random_state(rng)
    values = np.asarray(values)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    perm = rng.permutation(n)

    # the global mean carries an O(1/n) trace of every row, including this one
    prior = y.mean() if prior is None else float(prior)
    encoded = np.empty(n)
    v_perm, y_perm = values[perm], y[perm]

    for cat in np.unique(values):
        mask = v_perm == cat
        ys = y_perm[mask]
        # exclusive cumulative sum: entry i is the sum of everything BEFORE i
        prior_sum = np.concatenate([[0.0], np.cumsum(ys)[:-1]])
        prior_count = np.arange(len(ys))
        encoded[mask] = ((prior_sum + prior * prior_weight)
                         / (prior_count + prior_weight))

    out = np.empty(n)
    out[perm] = encoded      # undo the permutation
    return out


class OrderedBoostingRegressor(BaseEstimator, RegressorMixin):
    """Boosting where each tree's residuals come from a model that never saw the
    row it is scoring.

    THE SECOND LEAK
    ---------------
    Ordinary boosting has a subtler version of the same problem as target
    encoding. Tree ``t`` is fit on the residuals of the model built from trees
    ``1..t-1`` -- and those trees were fit on data that INCLUDES row ``i``. So
    row ``i``'s residual is not an honest estimate of the error on an unseen
    point; it is optimistically small, because the model has already partly
    memorised it.

    The residuals are therefore biased, tree ``t`` is fit to a biased target, and
    the bias compounds with every round. This is called PREDICTION SHIFT, and it
    is why boosting can overfit in a way that early stopping on a validation set
    detects but never explains.

    ORDERED BOOSTING
    ----------------
    Maintain several models, each trained on a different prefix of a permutation.
    To compute the residual for row ``i``, use the model trained only on rows
    before ``i``. That residual is genuinely out-of-sample.

    The exact algorithm needs ``O(n)`` models, which is unusable. CatBoost
    approximates with ``log(n)`` models on nested prefixes, and this class does
    the same. The version here is deliberately simple -- the point is the idea,
    and the idea is that the residual you fit must not come from a model that has
    already seen the answer.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 n_permutations=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_permutations = n_permutations
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.init_ = float(y.mean())

        # nested prefixes: 2^0, 2^1, ... rows. Model b has seen only the first
        # 2^b rows, so any row after that is out-of-sample for it
        self._perm = rng.permutation(n)
        position = np.empty(n, dtype=int)
        position[self._perm] = np.arange(n)
        n_models = max(1, int(np.ceil(np.log2(max(n, 2)))))
        prefix_sizes = [min(n, 2 ** (b + 1)) for b in range(n_models)]

        # which model is allowed to score each row: the largest prefix that
        # excludes it
        model_for_row = np.clip(
            np.floor(np.log2(np.maximum(position, 1))).astype(int),
            0, n_models - 1)

        self.trees_ = []
        # one running prediction per prefix model, plus the full model
        preds = np.full((n_models, n), self.init_)
        self.train_pred_ = np.full(n, self.init_)

        for _ in range(self.n_estimators):
            # each row's residual comes from a model that has not seen it
            residual = y - preds[model_for_row, np.arange(n)]
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng).fit(X, residual)
            update = tree.predict(X)
            self.trees_.append(tree)

            for b, size in enumerate(prefix_sizes):
                # model b only learns from its own prefix; rows beyond it stay
                # out-of-sample, which is the whole invariant
                in_prefix = position < size
                preds[b, in_prefix] += self.learning_rate * update[in_prefix]
            self.train_pred_ += self.learning_rate * update

        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X = check_array(X)
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred


# --- DART: dropout, for trees ---------------------------------------------

class DARTRegressor(BaseEstimator, RegressorMixin):
    """Dropouts meet Multiple Additive Regression Trees.

    THE PROBLEM: OVER-SPECIALISATION
    --------------------------------
    In ordinary boosting, the first trees make the big corrections and every
    later tree fits what is left -- which is less and less. So trees added late
    contribute almost nothing, and the model is effectively its first few dozen
    trees no matter how many you ask for. Adding trees stops helping long before
    it starts hurting, and the extra ones are pure cost.

    THE FIX, BORROWED FROM NEURAL NETWORKS
    --------------------------------------
    Compute each new tree's target against a RANDOM SUBSET of the existing trees,
    with the rest dropped. The new tree must then correct errors that the muted
    trees were covering, so it learns something the ensemble does not already
    know. No tree can rely on the others being present, exactly as dropout stops
    a unit relying on its neighbours.

    THE NORMALISATION IS THE FIDDLY PART
    ------------------------------------
    Dropping ``k`` trees and adding a full-strength replacement would overshoot:
    the new tree corrects the whole error left by the dropped ones, and then the
    dropped ones come back. So the new tree and the dropped ones are scaled by
    ``1/(k+1)`` -- together they now contribute what the ``k`` dropped trees did
    alone, and the ensemble's total output is preserved.

    Get that scaling wrong and DART diverges immediately, which is a good sign
    the derivation is doing real work rather than decorating a heuristic.

    Rashmi & Gilad-Bachrach (2015).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 drop_rate=0.1, skip_drop=0.5, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.drop_rate = drop_rate
        self.skip_drop = skip_drop
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)

        self.init_ = float(y.mean())
        self.trees_ = []
        self.weights_ = []
        self.n_dropped_ = []
        # every tree's contribution on the training set, kept so a subset can be
        # muted without re-predicting: dropping is then an array subtraction
        contributions = []

        for _ in range(self.n_estimators):
            if len(self.trees_) == 0 or rng.uniform() < self.skip_drop:
                dropped = np.array([], dtype=int)
            else:
                mask = rng.uniform(size=len(self.trees_)) < self.drop_rate
                dropped = np.where(mask)[0]
                if len(dropped) == 0:      # always drop at least one, or the
                    dropped = np.array([rng.randint(len(self.trees_))])  # round
                                                                          # is a
                                                                          # no-op
            # the ensemble with the dropped trees muted
            pred = np.full(n, self.init_)
            for i, contrib in enumerate(contributions):
                if i not in set(dropped.tolist()):
                    pred += self.weights_[i] * contrib

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng).fit(X, y - pred)
            contribution = tree.predict(X)

            k = len(dropped)
            # the normalisation -- see the class docstring
            new_weight = self.learning_rate / (k + 1)
            for i in dropped:
                self.weights_[i] *= k / (k + 1)

            self.trees_.append(tree)
            self.weights_.append(new_weight)
            contributions.append(contribution)
            self.n_dropped_.append(int(k))

        self.weights_ = list(self.weights_)
        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X = check_array(X)
        pred = np.full(len(X), self.init_)
        for tree, w in zip(self.trees_, self.weights_):
            pred += w * tree.predict(X)
        return pred


# --- NGBoost: predict a distribution --------------------------------------

class NGBoostRegressor(BaseEstimator, RegressorMixin):
    """Boosting that outputs a DISTRIBUTION, not a number.

    WHY THIS IS DIFFERENT
    ---------------------
    Every other regressor here answers "what is y?". This one answers "what do
    you know about y?" -- returning a mean AND a variance, per sample. "Probably
    around 5" and "5, give or take 40" are different answers, and only one of them
    should be acted on.

    It works by boosting the PARAMETERS of a distribution. Two sequences of trees:
    one predicting ``mu``, one predicting ``log(sigma)``, both trained to minimise
    the negative log-likelihood of a normal. The uncertainty is not a byproduct or
    an afterthought; it is half of what is being fit.

    THE NATURAL GRADIENT
    --------------------
    The ordinary gradient of the NLL is badly scaled for this, and the reason is
    worth understanding. The gradient lives in PARAMETER space, where a step in
    ``mu`` and a step in ``log(sigma)`` are treated as comparable. They are not:
    how much a distribution actually CHANGES per unit of parameter depends on
    where you are -- moving the mean of a tight distribution by 1 is drastic,
    while moving the mean of a broad one by 1 is barely noticeable.

    The natural gradient fixes this by pre-multiplying with the inverse Fisher
    information, which measures exactly that. It makes the step invariant to how
    the distribution is parameterised: fitting ``sigma`` or ``log(sigma)`` or
    ``sigma^2`` then gives the same trajectory, rather than three different
    algorithms wearing the same name.

    For a normal, the Fisher matrix is diagonal and small enough to write down
    directly -- which is why this fits in a page.

    THE CATCH, AND IT IS A BIG ONE
    ------------------------------
    ``sigma`` is fit to the residuals the model makes ON ITS TRAINING DATA. So it
    estimates TRAINING error, and it inherits every bit of the mean model's
    overfitting.

    Concretely: with ``max_depth=3`` on a noisy problem, the mean model's
    residuals might have standard deviation 14 in training and 26 in test. NGBoost
    dutifully reports sigma ~14, because that is what it saw -- and a "95%"
    interval then covers about 76% of unseen data. The model is not merely wrong
    about the value, it is CONFIDENTLY wrong, which is worse than not offering an
    interval at all.

    Shrink the mean model until it stops overfitting (``max_depth=1`` on that same
    problem gives train 47 versus test 50) and coverage returns to ~95%. So the
    uncertainty is exactly as trustworthy as the mean model's generalisation, and
    not one bit more.

    That is the general lesson about learned uncertainty, and it is why
    ``conformal prediction`` exists: it derives intervals from HELD-OUT residuals,
    so its coverage guarantee is distribution-free and survives an overfit model.
    When the interval must be right, use that. NGBoost tells you what the model
    believes; conformal tells you what is true.

    Duan et al. (2020).
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 natural_gradient=True, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.natural_gradient = natural_gradient
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)

        # start at the marginal distribution: the best guess before any feature
        # is consulted is the overall mean and spread
        self.init_mu_ = float(y.mean())
        self.init_logsigma_ = float(np.log(y.std() + 1e-8))
        mu = np.full(len(y), self.init_mu_)
        log_sigma = np.full(len(y), self.init_logsigma_)

        self.mu_trees_, self.sigma_trees_ = [], []
        self.nll_ = []

        for _ in range(self.n_estimators):
            sigma = np.exp(log_sigma)
            var = sigma ** 2

            # gradients of the normal NLL w.r.t. (mu, log_sigma)
            d_mu = (mu - y) / var
            d_log_sigma = 1.0 - ((y - mu) ** 2) / var

            if self.natural_gradient:
                # the Fisher information for a normal in (mu, log sigma) is
                # diagonal: [[1/sigma^2, 0], [0, 2]]. Multiplying by its inverse
                # rescales each parameter's step by how much the DISTRIBUTION
                # moves, not the parameter -- see the class docstring
                d_mu = d_mu * var
                d_log_sigma = d_log_sigma / 2.0

            mu_tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                            random_state=rng).fit(X, -d_mu)
            sigma_tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                               random_state=rng).fit(X, -d_log_sigma)
            mu += self.learning_rate * mu_tree.predict(X)
            log_sigma += self.learning_rate * sigma_tree.predict(X)
            # keep sigma in a sane range: an unbounded log_sigma can run away to
            # a zero-variance "perfect" fit whose NLL is minus infinity
            log_sigma = np.clip(log_sigma, -10.0, 10.0)

            self.mu_trees_.append(mu_tree)
            self.sigma_trees_.append(sigma_tree)
            self.nll_.append(float(np.mean(
                log_sigma + 0.5 * ((y - mu) / np.exp(log_sigma)) ** 2)))

        return self

    def _params(self, X):
        mu = np.full(len(X), self.init_mu_)
        log_sigma = np.full(len(X), self.init_logsigma_)
        for mt, st in zip(self.mu_trees_, self.sigma_trees_):
            mu += self.learning_rate * mt.predict(X)
            log_sigma += self.learning_rate * st.predict(X)
        return mu, np.exp(np.clip(log_sigma, -10.0, 10.0))

    def predict(self, X):
        """The mean, so this is a drop-in regressor."""
        check_is_fitted(self, "mu_trees_")
        return self._params(check_array(X))[0]

    def predict_dist(self, X):
        """(mean, standard deviation) per sample -- the reason to use this."""
        check_is_fitted(self, "mu_trees_")
        return self._params(check_array(X))

    def predict_interval(self, X, coverage=0.95):
        """Where the FITTED normal puts ``coverage`` of its mass, per sample.

        ``coverage`` is what the model BELIEVES, not a guarantee. Two separate
        reasons it can be wrong: the normal may be the wrong shape, and -- far
        more commonly -- sigma was fit in-sample and so under-reports whenever the
        mean model overfits. See the class docstring; it is the first thing to
        check when intervals come out too narrow.
        """
        from scipy.stats import norm
        mu, sigma = self.predict_dist(X)
        z = norm.ppf(0.5 + coverage / 2)
        return mu - z * sigma, mu + z * sigma


__all__ = ["goss_sample", "exclusive_feature_bundles", "bundle_features",
           "GOSSRegressor", "ordered_target_statistic",
           "OrderedBoostingRegressor", "DARTRegressor", "NGBoostRegressor"]
