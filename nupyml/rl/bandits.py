"""Multi-armed bandits: exploration vs exploitation, in its purest form.

THE SETUP
---------
``k`` arms, each paying a random reward from an unknown distribution. Pull one per
round; maximise total reward. There is no state and no delayed consequence, so
the ONLY difficulty is the exploration-exploitation trade -- which is exactly why
bandits are where you learn to reason about it.

The tension: to know an arm's value you must pull it (explore); to earn you must
pull the best one (exploit). Pull only the current best and you may never
discover a better arm you undersampled early. Explore forever and you waste pulls
on arms you already know are bad. REGRET -- the reward lost versus always pulling
the true best arm -- is how the trade is scored, and the methods below differ
entirely in how they manage it.

THE PROGRESSION
---------------
* ``EpsilonGreedy``    -- explore at random a fraction of the time. Simple,
  works, and dumb: it explores arms it already knows are terrible as eagerly as
  promising ones.
* ``UCB1``             -- "optimism under uncertainty". Explore arms whose value
  is UNCERTAIN, not random ones. Directed exploration, with a regret guarantee.
* ``ThompsonSampling`` -- keep a posterior over each arm's value; pull each in
  proportion to its probability of being best. Often the strongest in practice.
* ``LinUCB``           -- UCB when arms have FEATURES and reward is linear in
  them -- the contextual bandit, and the model behind news and ad selection.
* ``EXP3``             -- for an ADVERSARIAL bandit, where rewards are not random
  but chosen by an opponent. No statistics to trust, so it hedges.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class _BanditBase(BaseEstimator):
    def __init__(self, n_arms, random_state=None):
        self.n_arms = n_arms
        self.random_state = random_state

    def reset(self):
        self._rng = check_random_state(self.random_state)
        self.counts_ = np.zeros(self.n_arms)
        self.values_ = np.zeros(self.n_arms)   # running mean reward per arm
        self.t_ = 0
        return self

    def update(self, arm, reward):
        """Incremental mean: value += (reward - value) / count.

        The online update that never stores the history -- it is the sample mean
        rewritten as "nudge the estimate toward the new reward by 1/n". The same
        form recurs throughout RL as a fixed step size replaces 1/n.
        """
        self.counts_[arm] += 1
        self.t_ += 1
        self.values_[arm] += (reward - self.values_[arm]) / self.counts_[arm]
        return self


class EpsilonGreedy(_BanditBase):
    """Pull the best arm, except explore at random with probability epsilon.

    The baseline everyone starts from. Its flaw is that its exploration is
    UNDIRECTED: with probability epsilon it picks a uniformly random arm, giving a
    known-terrible arm exactly the same exploration budget as a promising one. The
    smarter methods below all fix this in different ways.

    ``epsilon`` can decay over time (``epsilon_decay``), which is usually right:
    explore hard early when you know nothing, exploit late once you do.
    """

    def __init__(self, n_arms, epsilon=0.1, epsilon_decay=1.0, random_state=None):
        super().__init__(n_arms, random_state)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select(self):
        if not hasattr(self, "counts_"):
            self.reset()
        eps = self.epsilon * (self.epsilon_decay ** self.t_)
        if self._rng.uniform() < eps:
            return self._rng.randint(self.n_arms)          # explore, blindly
        return int(np.argmax(self.values_))                # exploit


class UCB1(_BanditBase):
    """Optimism under uncertainty: pull the arm with the highest UPPER bound.

    THE IDEA
    --------
    Add to each arm's estimated value a bonus that grows with how UNCERTAIN that
    estimate is::

        score = mean_reward + c * sqrt(ln(t) / count)

    An arm pulled rarely has a large bonus, so it looks attractive and gets tried;
    each pull shrinks its bonus. Exploration is thus DIRECTED at the arms we know
    least about, not scattered at random -- the key improvement over epsilon-
    greedy.

    "Optimism" is exact: you act as if each arm is as good as its uncertainty
    allows. If the optimism is unwarranted, a few pulls reveal it and the bonus
    collapses; if warranted, you have found a good arm. Either outcome is
    progress, which is why UCB1's regret is provably logarithmic -- the best
    achievable.

    Auer, Cesa-Bianchi & Fischer (2002).
    """

    def __init__(self, n_arms, c=2.0, random_state=None):
        super().__init__(n_arms, random_state)
        self.c = c

    def select(self):
        if not hasattr(self, "counts_"):
            self.reset()
        # pull each arm once first: an unpulled arm has infinite uncertainty and
        # no mean to bound, so it must be sampled before the formula applies
        unpulled = np.where(self.counts_ == 0)[0]
        if len(unpulled):
            return int(unpulled[0])
        bonus = self.c * np.sqrt(np.log(self.t_) / self.counts_)
        return int(np.argmax(self.values_ + bonus))


class ThompsonSampling(_BanditBase):
    """Probability matching: pull each arm as often as it is probably the best.

    THE IDEA
    --------
    Keep a POSTERIOR distribution over each arm's true value. To choose, draw one
    sample from each arm's posterior and pull whichever sample is highest. An arm
    is thus pulled in proportion to its probability of being the best -- which is
    exactly the right amount of exploration, derived rather than tuned.

    A wide posterior (little data) sometimes samples high and gets explored;
    sometimes samples low and is skipped. As data accumulates the posterior
    narrows and the sampling converges on the true best arm. The randomness IS the
    exploration, and it self-calibrates -- there is no epsilon or c to set.

    This version assumes Bernoulli rewards with a Beta posterior, the textbook
    case: Beta is conjugate to the Bernoulli, so the update is just "increment
    alpha on a win, beta on a loss". Despite being the oldest bandit algorithm
    (Thompson, 1933) it is frequently the best in practice.
    """

    def __init__(self, n_arms, random_state=None):
        super().__init__(n_arms, random_state)

    def reset(self):
        super().reset()
        # Beta(1, 1) is uniform: before any data, every success rate is equally
        # plausible
        self.alpha_ = np.ones(self.n_arms)
        self.beta_ = np.ones(self.n_arms)
        return self

    def select(self):
        if not hasattr(self, "alpha_"):
            self.reset()
        # one draw from each arm's posterior; the winner is pulled
        samples = self._rng.beta(self.alpha_, self.beta_)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        super().update(arm, reward)
        # Beta is conjugate to Bernoulli: a win bumps alpha, a loss bumps beta
        self.alpha_[arm] += reward
        self.beta_[arm] += (1 - reward)
        return self


class LinUCB(_BanditBase):
    """Contextual bandit: UCB when the reward is linear in arm FEATURES.

    THE STEP UP FROM PLAIN BANDITS
    ------------------------------
    Real problems have context. Which article to show depends on the READER;
    which ad on the page. Plain bandits treat every round identically and cannot
    use it. LinUCB assumes each arm's expected reward is a linear function of a
    feature vector, ``reward ~ theta . x``, and learns ``theta`` by ridge
    regression -- online, as rewards arrive.

    The UCB idea carries straight over: pick the arm maximising predicted reward
    PLUS a bonus for uncertainty, where the uncertainty now comes from the ridge
    regression's covariance (how little it has seen in this feature direction).
    So it explores arms whose reward it cannot yet predict well GIVEN the current
    context -- optimism, lifted from scalars to a linear model.

    This is the algorithm behind large-scale news and ad recommendation.

    Li, Chu, Langford & Schapire (2010).
    """

    def __init__(self, n_arms, n_features, alpha=1.0, random_state=None):
        super().__init__(n_arms, random_state)
        self.n_features = n_features
        self.alpha = alpha

    def reset(self):
        self._rng = check_random_state(self.random_state)
        # per-arm ridge regression state: A is X^T X + I, b is X^T y
        self.A_ = [np.eye(self.n_features) for _ in range(self.n_arms)]
        self.b_ = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.t_ = 0
        return self

    def select(self, context):
        if not hasattr(self, "A_"):
            self.reset()
        x = np.asarray(context, float)
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A_[a])
            theta = A_inv @ self.b_[a]                 # ridge estimate for this arm
            mean = theta @ x
            # the bonus: sqrt(x^T A^-1 x) is the prediction's standard error in
            # this feature direction -- large where the arm has seen little like x
            bonus = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[a] = mean + bonus
        return int(np.argmax(scores))

    def update(self, arm, reward, context):
        x = np.asarray(context, float)
        # a rank-one update to the arm's ridge regression -- online least squares
        self.A_[arm] += np.outer(x, x)
        self.b_[arm] += reward * x
        self.t_ += 1
        return self


class EXP3(_BanditBase):
    """The ADVERSARIAL bandit: rewards chosen by an opponent, not by chance.

    WHY THE OTHER METHODS BREAK HERE
    --------------------------------
    UCB and Thompson assume each arm has a FIXED reward distribution to estimate.
    Drop that -- let an adversary set the rewards, possibly targeting whatever
    you seem to prefer -- and there is no distribution to be optimistic about. A
    method that trusts its estimates can be led anywhere.

    THE HEDGE
    ---------
    EXP3 keeps a WEIGHT per arm and samples in proportion, so it is never fully
    predictable -- an adversary cannot exploit a deterministic choice. Weights
    grow multiplicatively with reward (exponential weighting), and the crucial
    trick is IMPORTANCE WEIGHTING: divide an arm's observed reward by the
    probability it was played, so arms sampled rarely still get properly credited.
    Without it, seldom-pulled arms would be systematically underrated.

    It guarantees low regret against the best FIXED arm in hindsight, with no
    stochastic assumption at all -- the price of that robustness is being beaten
    by UCB when the world is in fact stochastic and benign.

    Auer, Cesa-Bianchi, Freund & Schapire (2002).
    """

    def __init__(self, n_arms, gamma=0.1, random_state=None):
        super().__init__(n_arms, random_state)
        self.gamma = gamma

    def reset(self):
        self._rng = check_random_state(self.random_state)
        self.weights_ = np.ones(self.n_arms)
        self.t_ = 0
        return self

    def _probs(self):
        w = self.weights_
        # mix the weighted distribution with a uniform floor: the gamma term
        # guarantees every arm keeps some probability, so nothing is ever
        # completely abandoned and importance weights stay bounded
        return (1 - self.gamma) * w / w.sum() + self.gamma / self.n_arms

    def select(self):
        if not hasattr(self, "weights_"):
            self.reset()
        self._p = self._probs()
        return int(self._rng.choice(self.n_arms, p=self._p))

    def update(self, arm, reward):
        # importance-weighted reward: credit the arm as if it had been played
        # 1/p of the time, so rare arms are not underrated
        estimated = reward / self._p[arm]
        self.weights_[arm] *= np.exp(self.gamma * estimated / self.n_arms)
        # renormalise to avoid overflow on long runs
        self.weights_ /= self.weights_.max()
        self.t_ += 1
        return self


__all__ = ["EpsilonGreedy", "UCB1", "ThompsonSampling", "LinUCB", "EXP3"]
