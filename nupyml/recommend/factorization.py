"""Matrix factorization and its relatives, for recommendation.

The training data is a list of ``(user, item, value)`` triples -- only the
OBSERVED cells, because the matrix is far too sparse to store densely and the
missing cells are the very thing to predict.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class MatrixFactorization(BaseEstimator):
    """Explicit-rating factorization by SGD: rating ~ user_vec . item_vec.

    THE MODEL
    ---------
    Each user ``u`` and item ``i`` gets a latent vector; the predicted rating is::

        r_hat = global_mean + user_bias[u] + item_bias[i] + p[u] . q[i]

    The BIASES matter as much as the dot product, which is the non-obvious part.
    Some users rate everything highly and some critically; some items are broadly
    loved. Those effects have nothing to do with the match between a user and an
    item, and folding them into the dot product would waste factors modelling
    "this user is generous". Separating them out lets the factors capture only the
    genuine INTERACTION -- who likes what -- which is the signal worth having.

    TRAINED ONLY ON OBSERVED CELLS
    ------------------------------
    The loss sums squared error over the ratings that EXIST, not over the full
    matrix. Treating missing as zero would teach the model that unrated means
    disliked, which is false -- unrated overwhelmingly means unseen. Iterating
    over the observed triples is what keeps that distinction; it is the single
    most important line in the fit.

    Regularisation (``reg``) shrinks the factors toward zero, without which a model
    with enough factors simply memorises the observed ratings and predicts noise
    for everything else.
    """

    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.1, n_epochs=50,
                 random_state=None):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, triples):
        rng = check_random_state(self.random_state)
        triples = [(int(u), int(i), float(r)) for u, i, r in triples]
        n_users = max(u for u, _, _ in triples) + 1
        n_items = max(i for _, i, _ in triples) + 1

        self.global_mean_ = np.mean([r for _, _, r in triples])
        # small random factors; zero would make every gradient identical and the
        # factors could never differentiate from each other
        self.P_ = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, size=(n_items, self.n_factors))
        self.user_bias_ = np.zeros(n_users)
        self.item_bias_ = np.zeros(n_items)
        self.history_ = []

        lr, reg = self.learning_rate, self.reg
        for _ in range(self.n_epochs):
            rng.shuffle(triples)
            sse = 0.0
            for u, i, r in triples:         # ONLY observed cells
                pred = (self.global_mean_ + self.user_bias_[u]
                        + self.item_bias_[i] + self.P_[u] @ self.Q_[i])
                err = r - pred
                sse += err ** 2
                # gradient step on this one rating, biases and factors together
                self.user_bias_[u] += lr * (err - reg * self.user_bias_[u])
                self.item_bias_[i] += lr * (err - reg * self.item_bias_[i])
                pu = self.P_[u].copy()
                self.P_[u] += lr * (err * self.Q_[i] - reg * self.P_[u])
                self.Q_[i] += lr * (err * pu - reg * self.Q_[i])
            self.history_.append(np.sqrt(sse / len(triples)))
        return self

    def predict(self, user, item):
        pred = (self.global_mean_ + self.user_bias_[user]
                + self.item_bias_[item] + self.P_[user] @ self.Q_[item])
        return float(pred)

    def recommend(self, user, n=10, exclude=None):
        """Top-``n`` items for a user by predicted rating."""
        scores = self.global_mean_ + self.user_bias_[user] + self.item_bias_ \
            + self.Q_ @ self.P_[user]
        if exclude is not None:
            scores[list(exclude)] = -np.inf
        return np.argsort(scores)[::-1][:n]


class ALS(BaseEstimator):
    """Alternating least squares: factorization by exact half-steps.

    THE INSIGHT
    -----------
    The objective is not convex in the user and item factors JOINTLY -- but FIX
    one and it becomes an ordinary least-squares problem in the other, which has a
    closed form. So alternate: solve exactly for all user factors with items
    fixed, then all item factors with users fixed, and repeat. Each half-step
    can only lower the loss, so it converges.

    WHY IT IS PREFERRED AT SCALE
    ----------------------------
    Within a half-step every user's factor solve is INDEPENDENT of every other
    user's, so the whole step parallelises trivially across a cluster -- which SGD
    on shuffled ratings does not. ALS is the classic choice for implicit-feedback
    factorization on huge matrices for exactly this reason. The trade is per-step
    cost: it solves a linear system per user/item rather than taking a cheap
    gradient step, so it wins on parallelism, not on single-machine speed.

    This version handles explicit ratings on a dense-ish matrix for clarity; the
    implicit-feedback weighting is the standard extension.
    """

    def __init__(self, n_factors=10, reg=0.1, n_iter=20, random_state=None):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, triples):
        rng = check_random_state(self.random_state)
        triples = [(int(u), int(i), float(r)) for u, i, r in triples]
        n_users = max(u for u, _, _ in triples) + 1
        n_items = max(i for _, i, _ in triples) + 1

        # rating matrix and a mask of which cells are observed
        R = np.zeros((n_users, n_items))
        M = np.zeros((n_users, n_items), dtype=bool)
        for u, i, r in triples:
            R[u, i] = r
            M[u, i] = True

        self.P_ = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, size=(n_items, self.n_factors))
        eye = self.reg * np.eye(self.n_factors)
        self.history_ = []

        for _ in range(self.n_iter):
            # fix Q, solve every user's factor by ridge regression on their
            # observed items -- a closed form, and independent across users
            for u in range(n_users):
                items = np.where(M[u])[0]
                if len(items) == 0:
                    continue
                Qi = self.Q_[items]
                self.P_[u] = np.linalg.solve(Qi.T @ Qi + eye, Qi.T @ R[u, items])
            # then fix P and solve every item's factor the same way
            for i in range(n_items):
                users = np.where(M[:, i])[0]
                if len(users) == 0:
                    continue
                Pu = self.P_[users]
                self.Q_[i] = np.linalg.solve(Pu.T @ Pu + eye, Pu.T @ R[users, i])
            pred = self.P_ @ self.Q_.T
            self.history_.append(np.sqrt(np.mean((pred[M] - R[M]) ** 2)))
        return self

    def predict(self, user, item):
        return float(self.P_[user] @ self.Q_[item])


class BPR(BaseEstimator):
    """Bayesian Personalized Ranking: factorization for IMPLICIT feedback.

    THE SHIFT FROM RATINGS TO RANKING
    ---------------------------------
    Clicks, plays and purchases are POSITIVE-ONLY: you see what a user engaged
    with, never what they rejected. A non-interaction is ambiguous -- unseen, not
    disliked. Fitting it as a "0 rating" (as rating models do) teaches the model
    that everything untouched is bad, which is wrong for most of the catalogue.

    BPR reframes the goal as RANKING: for a user, an item they interacted with
    should score HIGHER than one they did not. It maximises, over sampled triples
    ``(user, positive item, negative item)``::

        P(user prefers positive over negative) = sigmoid(score_pos - score_neg)

    So it never claims the negative is bad in absolute terms -- only that the
    positive is better, which is all the data actually supports. That relative
    objective is exactly right for implicit feedback, and getting it right (rather
    than regressing on clicks-as-ratings) is the difference between a recommender
    that works and one that recommends the already-popular.

    NEGATIVES ARE SAMPLED
    ---------------------
    There are far too many non-interactions to use them all, so each step draws a
    random item the user did NOT touch as the negative. Sampling makes the
    positive-only problem tractable, and is why BPR scales to huge implicit
    datasets.

    Rendle et al. (2009).
    """

    def __init__(self, n_factors=10, learning_rate=0.05, reg=0.01, n_epochs=50,
                 random_state=None):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, interactions):
        """``interactions`` is a list of ``(user, item)`` positive pairs."""
        rng = check_random_state(self.random_state)
        pairs = [(int(u), int(i)) for u, i in interactions]
        n_users = max(u for u, _ in pairs) + 1
        n_items = max(i for _, i in pairs) + 1

        # which items each user touched -- to sample negatives from the rest
        self.user_items_ = {}
        for u, i in pairs:
            self.user_items_.setdefault(u, set()).add(i)

        self.P_ = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, size=(n_items, self.n_factors))
        lr, reg = self.learning_rate, self.reg

        for _ in range(self.n_epochs):
            rng.shuffle(pairs)
            for u, pos in pairs:
                # sample a negative the user has NOT interacted with
                neg = rng.randint(n_items)
                while neg in self.user_items_[u]:
                    neg = rng.randint(n_items)
                # the ranking gradient: push the positive above the negative
                x_uij = self.P_[u] @ (self.Q_[pos] - self.Q_[neg])
                sig = 1.0 / (1.0 + np.exp(x_uij))     # = 1 - sigmoid(x_uij)
                pu = self.P_[u].copy()
                self.P_[u] += lr * (sig * (self.Q_[pos] - self.Q_[neg])
                                    - reg * self.P_[u])
                self.Q_[pos] += lr * (sig * pu - reg * self.Q_[pos])
                self.Q_[neg] += lr * (-sig * pu - reg * self.Q_[neg])
        return self

    def score(self, user, item):
        return float(self.P_[user] @ self.Q_[item])

    def recommend(self, user, n=10):
        scores = self.Q_ @ self.P_[user]
        # never recommend what the user already has -- the standard filter
        for i in self.user_items_.get(user, ()):
            scores[i] = -np.inf
        return np.argsort(scores)[::-1][:n]


class FactorizationMachine(BaseEstimator):
    """Factorization generalised to ARBITRARY features, with all pairwise terms.

    THE GENERALISATION
    ------------------
    Matrix factorization models one interaction -- user x item. A factorization
    machine models ALL pairwise feature interactions::

        y = w0 + sum_i w_i x_i + sum_{i<j} (v_i . v_j) x_i x_j

    A plain quadratic model would need one parameter per feature PAIR -- ``d^2`` of
    them, unlearnable when the features are sparse (most pairs are never even
    observed together). The FM instead gives each feature a latent VECTOR and
    models a pair's weight as the DOT PRODUCT of the two vectors. Now ``O(d*k)``
    parameters cover all pairs, and a pair never seen in training still gets a
    sensible weight, borrowed from each feature's behaviour in the pairs that WERE
    seen. That factorised interaction weight is the whole idea.

    THE O(n) TRICK
    --------------
    Evaluating all pairwise terms looks like ``O(d^2)``, but the sum has a closed
    form -- ``0.5 * (square of the sum minus sum of the squares)`` of the ``v*x``
    vectors -- computable in ``O(d*k)``. Without that identity FMs would be as slow
    as the explicit quadratic model they replace; with it they are linear in the
    number of features. Encode user-id and item-id as one-hot features and an FM
    reduces exactly to biased matrix factorization -- which is why it is the
    bridge from recommendation to general sparse prediction.

    Rendle (2010).
    """

    def __init__(self, n_factors=8, learning_rate=0.01, reg=0.01, n_epochs=50,
                 task="regression", random_state=None):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.task = task
        self.random_state = random_state

    def _predict_raw(self, x):
        linear = self.w0_ + self.w_ @ x
        # the O(d*k) form of the pairwise sum -- see the class docstring
        vx = self.V_.T @ x                       # (k,)
        v2x2 = (self.V_ ** 2).T @ (x ** 2)       # (k,)
        interaction = 0.5 * np.sum(vx ** 2 - v2x2)
        return linear + interaction

    def fit(self, X, y):
        from ..utils import check_X_y
        X, y = check_X_y(X, y, y_numeric=(self.task == "regression"))
        rng = check_random_state(self.random_state)
        n, d = X.shape
        self.w0_ = 0.0
        self.w_ = np.zeros(d)
        self.V_ = rng.normal(0, 0.1, size=(d, self.n_factors))
        lr, reg = self.learning_rate, self.reg
        y = np.asarray(y, float)

        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for idx in order:
                x = X[idx]
                pred = self._predict_raw(x)
                if self.task == "classification":
                    p = 1.0 / (1.0 + np.exp(-np.clip(pred, -30, 30)))
                    err = p - y[idx]
                else:
                    err = pred - y[idx]
                self.w0_ -= lr * err
                self.w_ -= lr * (err * x + reg * self.w_)
                # the factor gradient uses the precomputed vx term
                vx = self.V_.T @ x
                for f in range(self.n_factors):
                    grad = x * (vx[f] - self.V_[:, f] * x)
                    self.V_[:, f] -= lr * (err * grad + reg * self.V_[:, f])
        return self

    def predict(self, X):
        from ..utils import check_array
        X = check_array(X)
        raw = np.array([self._predict_raw(x) for x in X])
        if self.task == "classification":
            return (raw > 0).astype(int)
        return raw

    def predict_proba(self, X):
        from ..utils import check_array
        X = check_array(X)
        raw = np.array([self._predict_raw(x) for x in X])
        p = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        return np.column_stack([1 - p, p])


__all__ = ["MatrixFactorization", "ALS", "BPR", "FactorizationMachine"]
