"""Evolutionary search wearing sklearn's clothes.

Two places where the derivative-free family earns its cost on ordinary ML
problems, because the objective genuinely has no gradient:

* **Feature selection** is a subset choice -- ``2^n`` options, and no derivative
  with respect to "include feature 3".
* **Hyperparameter search** optimizes cross-validated score, which is piecewise
  constant in the parameters and evaluated through a whole training run. Nothing
  to differentiate, and every evaluation is expensive.

A NOTE ON THE SIGN
------------------
The optimizers in this package minimise; sklearn scores are maximised. These
wrappers negate internally so that ``best_score_`` is a SCORE (higher is better)
as an sklearn user expects, while the search underneath still minimises. The
convention change happens here, at the boundary, and nowhere else.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, clone, check_is_fitted
from ..model_selection import cross_val_score, KFold
from ..utils import check_array, check_random_state
from .genetic import BinaryGeneticAlgorithm


class GAFeatureSelector(BaseEstimator, TransformerMixin):
    """Choose a feature subset with a genetic algorithm.

    WHY NOT JUST RANK THE FEATURES?
    -------------------------------
    Univariate selection scores each feature alone and keeps the top k. It cannot
    see interactions, and they are common:

    * two features may be useless individually and perfectly predictive together
      (XOR is the standard example -- each input alone is pure noise);
    * two excellent features may be near-duplicates, so keeping both buys almost
      nothing over keeping one.

    A GA evaluates SUBSETS, so both effects are visible to it. That is the case
    for spending far more compute than a ranking would cost.

    THE HAZARD
    ----------
    This searches subsets against a cross-validated score, and it searches HARD.
    With enough generations it will find a subset that fits the CV folds' noise
    -- the selection itself overfits, even though each individual evaluation is
    honest. The resulting score is optimistically biased and is not an estimate
    of anything. Evaluate the chosen subset on data that this search never saw.

    ``size_penalty`` subtracts ``size_penalty * n_selected`` from the score, which
    breaks ties toward smaller subsets and works against the drift toward
    including everything.
    """

    def __init__(self, estimator, cv=3, scoring=None, population_size=30,
                 n_generations=20, size_penalty=0.001, tournament_k=3,
                 random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.n_generations = n_generations
        self.size_penalty = size_penalty
        self.tournament_k = tournament_k
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)

        def objective(mask):
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                return 1e10        # the empty subset is not a candidate
            score = cross_val_score(clone(self.estimator), X[:, mask], y,
                                    cv=self.cv, scoring=self.scoring).mean()
            # negated: the GA minimises, sklearn scores maximise
            return -score + self.size_penalty * mask.sum()

        ga = BinaryGeneticAlgorithm(
            objective, X.shape[1], population_size=self.population_size,
            n_generations=self.n_generations, tournament_k=self.tournament_k,
            random_state=rng).run()

        self.support_ = np.asarray(ga.best_, dtype=bool)
        self.n_features_out_ = int(self.support_.sum())
        # reported as a score, with the penalty removed: the penalty is a search
        # device, not part of what the subset actually achieves
        self.best_score_ = float(-ga.best_fitness_
                                 + self.size_penalty * self.support_.sum())
        self.history_ = ga.history_
        self.estimator_ = clone(self.estimator).fit(X[:, self.support_], y)
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return check_array(X)[:, self.support_]

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.transform(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            return np.array([f"x{i}" for i in np.where(self.support_)[0]])
        return np.asarray(input_features)[self.support_]


class GASearchCV(BaseEstimator):
    """Hyperparameter search by genetic algorithm.

    WHERE THIS BEATS THE ALTERNATIVES
    ---------------------------------
    * **Grid search** costs the product of every axis, so it dies at three or
      four parameters -- and it spends that budget on combinations it could
      already tell were hopeless.
    * **Random search** beats grid search (Bergstra & Bengio) because most
      parameters do not matter, and random sampling spends its budget on the ones
      that do. But it never learns: sample 900 is drawn exactly as blindly as
      sample 1.
    * A **GA** concentrates its later evaluations near what has already worked.
      With a real budget on a rugged landscape that pays; with a small budget, or
      few parameters, random search is a better use of the same evaluations and
      is far simpler.

    Bayesian optimization is the stronger answer when evaluations are the
    bottleneck, since it MODELS the objective rather than merely remembering good
    points. It arrives in a later phase of this library.

    ``param_space`` maps a name to either ``(lo, hi)`` for a continuous parameter
    or a list of discrete choices. Discrete parameters are searched by evolving a
    real value and rounding it to an index -- a small dishonesty that keeps one
    representation for both kinds, at the cost of imposing an ordering on
    categories that may have none.
    """

    def __init__(self, estimator, param_space, cv=3, scoring=None,
                 population_size=20, n_generations=15, random_state=None):
        self.estimator = estimator
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.n_generations = n_generations
        self.random_state = random_state

    def _decode(self, vector):
        """Turn a real-valued individual into a parameter dict."""
        params = {}
        for value, (name, space) in zip(vector, self.param_space.items()):
            if isinstance(space, tuple) and len(space) == 2 and \
                    all(isinstance(v, (int, float)) for v in space):
                params[name] = float(value)
            else:
                # a discrete axis: clamp to a valid index -- see the class docstring
                idx = int(np.clip(round(value), 0, len(space) - 1))
                params[name] = space[idx]
        return params

    def _bounds(self):
        bounds = []
        for space in self.param_space.values():
            if isinstance(space, tuple) and len(space) == 2 and \
                    all(isinstance(v, (int, float)) for v in space):
                bounds.append(space)
            else:
                bounds.append((0, len(space) - 1))
        return np.array(bounds, dtype=np.float64)

    def fit(self, X, y):
        from .genetic import GeneticAlgorithm
        X = check_array(X)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        seen = {}

        def objective(vector):
            params = self._decode(vector)
            key = tuple(sorted((k, round(v, 6) if isinstance(v, float) else v)
                               for k, v in params.items()))
            # a GA revisits: elites persist and children resemble parents, so
            # without a cache the same fit is paid for repeatedly, and a fit is
            # the expensive thing here
            if key in seen:
                return seen[key]
            est = clone(self.estimator).set_params(**params)
            try:
                score = cross_val_score(est, X, y, cv=self.cv,
                                        scoring=self.scoring).mean()
            except Exception:
                # an invalid combination is a normal event in a stochastic
                # search, not an error: score it as terrible and move on
                score = -1e10
            seen[key] = -score      # negated: the GA minimises
            return seen[key]

        ga = GeneticAlgorithm(
            objective, self._bounds(), population_size=self.population_size,
            n_generations=self.n_generations, random_state=rng).run()

        self.best_params_ = self._decode(ga.best_)
        self.best_score_ = float(-ga.best_fitness_)
        self.history_ = [-h for h in ga.history_]
        self.n_evaluations_ = len(seen)
        self.best_estimator_ = clone(self.estimator).set_params(
            **self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.score(X, y)


__all__ = ["GAFeatureSelector", "GASearchCV"]
