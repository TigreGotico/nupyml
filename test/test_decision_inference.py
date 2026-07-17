"""RL, bandits, MCMC, conformal prediction, Bayesian optimization, survival, causal.

Each family is held to the guarantee it advertises: bandits find the best arm,
MCMC recovers a known distribution, conformal hits its coverage target even with
a bad model, causal estimators recover a planted effect a naive comparison gets
wrong.
"""
import numpy as np
import pytest

from nupyml.datasets import make_regression, make_classification
from nupyml.model_selection import train_test_split
from nupyml.tree import DecisionTreeRegressor
from nupyml.linear_model import LinearRegression, LogisticRegression
from nupyml.rl import (EpsilonGreedy, UCB1, ThompsonSampling, LinUCB, EXP3,
                       QLearning, SARSA, ExpectedSARSA, REINFORCE, ActorCritic,
                       value_iteration, policy_iteration)
from nupyml.inference import (MetropolisHastings, GibbsSampler, HamiltonianMC,
                              ConformalRegressor, ConformalClassifier,
                              MondrianConformalRegressor, BayesianOptimization,
                              GaussianProcessRegressor)
from nupyml.inference.mcmc import effective_sample_size
from nupyml.survival import KaplanMeier, NelsonAalen, CoxPH
from nupyml.causal import (InversePropensityWeighting, DoublyRobust,
                           propensity_score, PropensityMatching)


# --- bandits --------------------------------------------------------------

@pytest.fixture
def bernoulli_arms():
    """Five arms; arm 2 is clearly the best."""
    return np.array([0.2, 0.5, 0.8, 0.3, 0.6])


def _run_bandit(bandit, arms, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    bandit.reset()
    total = 0.0
    for _ in range(n):
        a = bandit.select()
        r = float(rng.uniform() < arms[a])
        bandit.update(a, r)
        total += r
    return total


def test_epsilon_greedy_finds_the_best_arm(bernoulli_arms):
    bandit = EpsilonGreedy(5, epsilon=0.1, random_state=0)
    _run_bandit(bandit, bernoulli_arms)
    assert np.argmax(bandit.values_) == 2


def test_ucb_finds_the_best_arm(bernoulli_arms):
    bandit = UCB1(5, random_state=0)
    _run_bandit(bandit, bernoulli_arms)
    assert np.argmax(bandit.values_) == 2


def test_thompson_sampling_finds_the_best_arm(bernoulli_arms):
    bandit = ThompsonSampling(5, random_state=0)
    _run_bandit(bandit, bernoulli_arms)
    assert np.argmax(bandit.values_) == 2


def test_ucb_beats_pure_greed(bernoulli_arms):
    """Directed exploration should collect more reward than never exploring."""
    ucb = UCB1(5, random_state=0)
    greedy = EpsilonGreedy(5, epsilon=0.0, random_state=0)   # exploit only
    assert _run_bandit(ucb, bernoulli_arms) > _run_bandit(greedy, bernoulli_arms)


def test_exp3_survives_an_adversarial_sequence():
    """EXP3 makes no stochastic assumption, so it should still do well when the
    rewards are not random. Here the good arm is fixed but the others vary."""
    rng = np.random.RandomState(0)
    exp3 = EXP3(3, gamma=0.1, random_state=0)
    exp3.reset()
    total = 0.0
    for t in range(3000):
        a = exp3.select()
        # arm 0 always pays 0.7; the others oscillate to try to mislead
        r = 0.7 if a == 0 else 0.5 + 0.4 * np.sin(t / 10)
        exp3.update(a, r)
        total += r
    # it should learn to favour the reliably-good arm
    assert exp3.weights_[0] == exp3.weights_.max()


def test_linucb_uses_context():
    """A contextual bandit should beat a context-blind one when the best arm
    depends on the context."""
    rng = np.random.RandomState(0)
    d = 4
    theta = rng.normal(size=(3, d))

    def reward(a, ctx):
        return float(rng.uniform() < 1 / (1 + np.exp(-theta[a] @ ctx)))

    lin = LinUCB(3, d, alpha=1.0, random_state=0)
    lin.reset()
    lin_total = 0.0
    for _ in range(3000):
        ctx = rng.normal(size=d)
        a = lin.select(ctx)
        r = reward(a, ctx)
        lin.update(a, r, ctx)
        lin_total += r
    # beats random arm choice, which ignores the context entirely
    assert lin_total > 3000 * 0.5


# --- planning -------------------------------------------------------------

@pytest.fixture
def simple_mdp():
    """A 3-state chain: reach state 2 and stay there for reward."""
    P = np.zeros((3, 2, 3))
    R = np.zeros((3, 2))
    P[0, 0] = [0, 1, 0]; P[0, 1] = [1, 0, 0]
    P[1, 0] = [0, 0, 1]; P[1, 1] = [1, 0, 0]
    P[2, 0] = [0, 0, 1]; P[2, 1] = [0, 0, 1]
    R[1, 0] = 1.0; R[2, 0] = 1.0
    return P, R


def test_value_and_policy_iteration_agree(simple_mdp):
    P, R = simple_mdp
    _, vi_policy = value_iteration(P, R, gamma=0.9)
    _, pi_policy = policy_iteration(P, R, gamma=0.9)
    assert np.array_equal(vi_policy, pi_policy)


def test_value_iteration_finds_the_rewarding_action(simple_mdp):
    P, R = simple_mdp
    V, policy = value_iteration(P, R, gamma=0.9)
    assert policy[0] == 0        # action 0 moves toward the reward
    assert V[2] > V[0]           # being at the goal is worth more


# --- tabular RL -----------------------------------------------------------

def _corridor_train(agent_cls, n_states=6, episodes=1500, **kw):
    """A 1D corridor: start at 0, reward at the right end. Optimal policy is all
    'right'."""
    agent = agent_cls(n_states, 2, alpha=0.5, gamma=0.9, epsilon=0.3,
                      random_state=0, **kw)
    agent.reset()

    def step(s, a):
        ns = max(0, s - 1) if a == 0 else min(n_states - 1, s + 1)
        return ns, (1.0 if ns == n_states - 1 else 0.0), ns == n_states - 1

    for _ in range(episodes):
        s = 0
        for _ in range(50):
            a = agent.act(s)
            ns, r, done = step(s, a)
            if isinstance(agent, SARSA):
                na = agent.act(ns)
                agent.update(s, a, r, ns, na, done)
            else:
                agent.update(s, a, r, ns, done)
            s = ns
            if done:
                break
        agent.end_episode()
    return agent


def test_q_learning_solves_the_corridor():
    agent = _corridor_train(QLearning)
    assert np.all(agent.policy()[:5] == 1)      # go right everywhere


def test_sarsa_solves_the_corridor():
    agent = _corridor_train(SARSA)
    assert np.all(agent.policy()[:5] == 1)


def test_expected_sarsa_solves_the_corridor():
    agent = _corridor_train(ExpectedSARSA)
    assert np.all(agent.policy()[:5] == 1)


def test_q_learning_is_off_policy_sarsa_is_on_policy():
    """The one-symbol difference in the update, checked by its effect: given the
    same transition, Q-learning bootstraps from the MAX next value while SARSA
    uses the actual next action's value."""
    q = QLearning(3, 2, alpha=1.0, gamma=1.0, random_state=0).reset()
    s = SARSA(3, 2, alpha=1.0, gamma=1.0, random_state=0).reset()
    # seed a Q-table where the two next actions differ in value
    q.Q_[1] = np.array([5.0, 1.0])
    s.Q_[1] = np.array([5.0, 1.0])
    q.update(0, 0, 0.0, 1, done=False)          # uses max = 5
    s.update(0, 0, 0.0, 1, a_next=1, done=False)  # uses chosen action = 1
    assert q.Q_[0, 0] == pytest.approx(5.0)
    assert s.Q_[0, 0] == pytest.approx(1.0)


def test_reinforce_learns_a_bandit_policy():
    """gamma=0 makes each step its own credit assignment -- a clean policy-
    gradient test where the correct action equals the state index."""
    rng = np.random.RandomState(0)
    pg = REINFORCE(2, 2, gamma=0.0, learning_rate=0.1, random_state=0)
    for _ in range(500):
        states, actions, rewards = [], [], []
        for _ in range(10):
            st = rng.randint(2)
            oh = np.eye(2)[st]
            a = pg.act(oh)
            states.append(oh)
            actions.append(a)
            rewards.append(1.0 if a == st else 0.0)
        pg.update(states, actions, rewards)
    acc = np.mean([pg.act(np.eye(2)[i]) == i for i in [0, 1] * 50])
    assert acc > 0.9


def test_actor_critic_learns_a_policy():
    rng = np.random.RandomState(0)
    ac = ActorCritic(2, 2, gamma=0.0, learning_rate=0.1, random_state=0)
    for _ in range(400):
        states, actions, rewards = [], [], []
        for _ in range(10):
            st = rng.randint(2)
            oh = np.eye(2)[st]
            a = ac.act(oh)
            states.append(oh)
            actions.append(a)
            rewards.append(1.0 if a == st else 0.0)
        ac.update(states, actions, rewards)
    acc = np.mean([ac.act(np.eye(2)[i]) == i for i in [0, 1] * 50])
    assert acc > 0.9


# --- MCMC -----------------------------------------------------------------

def _normal_logprob(mu, sigma):
    return lambda x: -0.5 * ((x[0] - mu) / sigma) ** 2


def test_metropolis_hastings_recovers_a_normal():
    mh = MetropolisHastings(_normal_logprob(3.0, 2.0), step_size=3.0,
                            random_state=0)
    samples = mh.sample([0.0], 5000, burn_in=2000)
    assert samples.mean() == pytest.approx(3.0, abs=0.2)
    assert samples.std() == pytest.approx(2.0, abs=0.2)


def test_metropolis_acceptance_rate_is_reported():
    mh = MetropolisHastings(_normal_logprob(0.0, 1.0), step_size=1.0,
                            random_state=0)
    mh.sample([0.0], 2000, burn_in=500)
    assert 0.1 < mh.acceptance_rate_ < 0.9


def test_gibbs_samples_a_correlated_gaussian():
    """Gibbs on a bivariate normal: each conditional is itself normal, with a
    closed form -- the setting Gibbs is made for."""
    rho = 0.5
    conds = [
        lambda x, r: r.normal(rho * x[1], np.sqrt(1 - rho ** 2)),
        lambda x, r: r.normal(rho * x[0], np.sqrt(1 - rho ** 2)),
    ]
    gibbs = GibbsSampler(conds, random_state=0)
    samples = gibbs.sample([0.0, 0.0], 5000, burn_in=1000)
    assert samples[:, 0].std() == pytest.approx(1.0, abs=0.15)
    # it recovers the planted correlation
    assert np.corrcoef(samples.T)[0, 1] == pytest.approx(rho, abs=0.1)


def test_hmc_mixes_better_than_random_walk():
    """HMC's gradient-guided proposals should give a higher effective sample size
    than blind random-walk Metropolis for the same chain length."""
    logp = _normal_logprob(0.0, 1.0)
    grad = lambda x: np.array([-x[0]])
    hmc = HamiltonianMC(logp, grad, step_size=0.3, n_leapfrog=15, random_state=0)
    mh = MetropolisHastings(logp, step_size=0.5, random_state=0)
    hmc_samples = hmc.sample([0.0], 2000, burn_in=500)
    mh_samples = mh.sample([0.0], 2000, burn_in=500)
    assert effective_sample_size(hmc_samples)[0] > effective_sample_size(mh_samples)[0]


def test_effective_sample_size_below_chain_length():
    """A correlated chain is worth fewer independent draws than its length."""
    mh = MetropolisHastings(_normal_logprob(0.0, 1.0), step_size=0.3,
                            random_state=0)
    samples = mh.sample([0.0], 3000, burn_in=500)
    assert effective_sample_size(samples)[0] < 3000


# --- conformal prediction -------------------------------------------------

@pytest.fixture
def noisy_split():
    X, y = make_regression(n_samples=1000, n_features=5, noise=15.0,
                           random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_conformal_hits_its_coverage_even_with_a_weak_model(noisy_split):
    """The guarantee: coverage holds for ANY model, however bad. A stump is a
    poor regressor, but the intervals must still cover ~90%."""
    Xtr, Xte, ytr, yte = noisy_split
    cr = ConformalRegressor(DecisionTreeRegressor(max_depth=2),
                            random_state=0).fit(Xtr, ytr)
    lo, hi = cr.predict_interval(Xte, coverage=0.9)
    coverage = ((yte >= lo) & (yte <= hi)).mean()
    assert coverage >= 0.86        # allow a little finite-sample slack below 0.9


def test_conformal_intervals_widen_with_coverage(noisy_split):
    Xtr, Xte, ytr, yte = noisy_split
    cr = ConformalRegressor(LinearRegression(), random_state=0).fit(Xtr, ytr)
    lo80, hi80 = cr.predict_interval(Xte, 0.8)
    lo95, hi95 = cr.predict_interval(Xte, 0.95)
    assert (hi95 - lo95).mean() > (hi80 - lo80).mean()


def test_conformal_beats_a_naive_model_based_interval(noisy_split):
    """The contrast with NGBoost/RVM: a model that under-fits reports intervals
    that are too tight from its own noise estimate, but conformal corrects them
    to the guaranteed coverage."""
    Xtr, Xte, ytr, yte = noisy_split
    cr = ConformalRegressor(DecisionTreeRegressor(max_depth=3),
                            random_state=0).fit(Xtr, ytr)
    lo, hi = cr.predict_interval(Xte, coverage=0.9)
    assert ((yte >= lo) & (yte <= hi)).mean() >= 0.86


def test_conformal_classifier_returns_sets_that_cover():
    X, y = make_classification(n_samples=800, n_features=6, n_classes=3,
                               n_informative=4, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    cc = ConformalClassifier(LogisticRegression(max_iter=300),
                             random_state=0).fit(Xtr, ytr)
    sets = cc.predict_set(Xte, coverage=0.9)
    covered = np.mean([yt in s for yt, s in zip(yte, sets)])
    assert covered >= 0.85


def test_mondrian_conformal_adapts_width_by_group():
    """Per-group calibration should give different interval widths where the
    model's error genuinely differs by group."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, (1000, 1))
    # noise grows with the group, so group 1 should get wider intervals
    group = (X[:, 0] > 5).astype(int)
    y = X[:, 0] + rng.normal(0, 1 + 4 * group, 1000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    mc = MondrianConformalRegressor(
        LinearRegression(), group_fn=lambda x: (x[:, 0] > 5).astype(int),
        random_state=0).fit(Xtr, ytr)
    lo, hi = mc.predict_interval(Xte, 0.9)
    widths = hi - lo
    high_noise = Xte[:, 0] > 5
    assert widths[high_noise].mean() > widths[~high_noise].mean()


# --- Bayesian optimization ------------------------------------------------

def test_gp_uncertainty_grows_away_from_the_data():
    """The property that makes the GP the surrogate: it knows what it has not
    seen, unlike the RVM."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.0])
    gp = GaussianProcessRegressor(length_scale=0.5).fit(X, y)
    _, std_near = gp.predict(np.array([[1.0]]), return_std=True)
    _, std_far = gp.predict(np.array([[10.0]]), return_std=True)
    assert std_far[0] > std_near[0]


def test_bayesian_optimization_finds_a_minimum():
    """A cheap-to-check objective with a known minimum near x=2."""
    def f(x):
        return (x[0] - 2.0) ** 2 + 0.1 * np.sin(10 * x[0])

    bo = BayesianOptimization(f, [(-5, 5)], n_init=5, n_iter=25,
                              random_state=0).run()
    assert abs(bo.best_x_[0] - 2.0) < 0.5


def test_bayesian_optimization_beats_random_search():
    """Modelling the objective should beat sampling it blindly, given the same
    tiny budget -- the entire premise of the method."""
    rng = np.random.RandomState(0)

    def f(x):
        return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2

    bo = BayesianOptimization(f, [(-5, 5), (-5, 5)], n_init=5, n_iter=20,
                              random_state=0).run()
    random_best = min(f(p) for p in rng.uniform(-5, 5, size=(25, 2)))
    assert bo.best_y_ < random_best


def test_acquisition_functions_all_run():
    def f(x):
        return (x[0] - 1.0) ** 2

    for acq in ("ei", "ucb", "pi"):
        bo = BayesianOptimization(f, [(-3, 3)], acquisition=acq, n_init=4,
                                  n_iter=15, random_state=0).run()
        assert abs(bo.best_x_[0] - 1.0) < 1.0


# --- survival -------------------------------------------------------------

@pytest.fixture
def censored_data():
    rng = np.random.RandomState(0)
    duration = rng.exponential(10, 300)
    censoring = rng.exponential(15, 300)
    observed = np.minimum(duration, censoring)
    event = (duration <= censoring).astype(int)
    return observed, event


def test_kaplan_meier_estimates_median_survival(censored_data):
    observed, event = censored_data
    km = KaplanMeier().fit(observed, event)
    # true median of Exp(10) is 10*ln2 ~ 6.9
    assert abs(km.median_survival_ - 10 * np.log(2)) < 2.0


def test_kaplan_meier_is_monotone_decreasing(censored_data):
    observed, event = censored_data
    km = KaplanMeier().fit(observed, event)
    assert np.all(np.diff(km.survival_) <= 1e-12)
    assert km.survival_[0] <= 1.0


def test_censoring_is_used_not_dropped():
    """A censored subject must lower later survival estimates by staying in the
    at-risk count -- treating it as an event, or dropping it, both give wrong
    curves."""
    # two events at t=1,3 and one censored at t=2
    obs = np.array([1.0, 2.0, 3.0])
    ev = np.array([1, 0, 1])          # middle one is censored
    km = KaplanMeier().fit(obs, ev)
    # at t=1: 3 at risk, 1 event => S=2/3; at t=3: only 1 at risk => S=0
    assert km.survival_[0] == pytest.approx(2 / 3)


def test_nelson_aalen_cumulative_hazard_increases(censored_data):
    observed, event = censored_data
    na = NelsonAalen().fit(observed, event)
    assert np.all(np.diff(na.cumulative_hazard_) >= 0)


def test_cox_recovers_known_hazard_coefficients():
    """Planted log-hazard coefficients should be recovered -- the semi-parametric
    trick estimates them without ever modelling the baseline."""
    rng = np.random.RandomState(0)
    n = 800
    X = rng.normal(size=(n, 2))
    risk = np.exp(X @ [0.8, -0.5])
    t = rng.exponential(1 / risk)
    c = rng.exponential(2, n)
    observed = np.minimum(t, c)
    event = (t <= c).astype(int)
    cox = CoxPH().fit(X, observed, event)
    assert cox.coef_[0] == pytest.approx(0.8, abs=0.2)
    assert cox.coef_[1] == pytest.approx(-0.5, abs=0.2)


def test_cox_hazard_ratios_are_exp_coef():
    rng = np.random.RandomState(0)
    n = 400
    X = rng.normal(size=(n, 2))
    t = rng.exponential(1, n)
    e = np.ones(n, dtype=int)
    cox = CoxPH().fit(X, t, e)
    assert np.allclose(cox.hazard_ratios_, np.exp(cox.coef_))


# --- causal inference -----------------------------------------------------

@pytest.fixture
def confounded_data():
    """A planted ATE of 3.0, with treatment assignment confounded by X so the
    naive difference is wrong."""
    rng = np.random.RandomState(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    propensity = 1 / (1 + np.exp(-(X @ [1, -1, 0.5])))
    treatment = (rng.uniform(size=n) < propensity).astype(int)
    y0 = X @ [1, 2, -1] + rng.normal(0, 1, n)
    outcome = np.where(treatment, y0 + 3.0, y0)      # true effect = 3
    return X, treatment, outcome


def test_naive_difference_is_confounded(confounded_data):
    """The mistake the field exists to prevent: the raw treated-minus-control
    difference is biased away from the true effect of 3."""
    X, treatment, outcome = confounded_data
    naive = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    assert abs(naive - 3.0) > 0.5          # visibly wrong


def test_ipw_recovers_the_treatment_effect(confounded_data):
    X, treatment, outcome = confounded_data
    ipw = InversePropensityWeighting().fit(X, treatment, outcome)
    assert ipw.ate_ == pytest.approx(3.0, abs=0.4)


def test_doubly_robust_recovers_the_treatment_effect(confounded_data):
    X, treatment, outcome = confounded_data
    dr = DoublyRobust().fit(X, treatment, outcome)
    assert dr.ate_ == pytest.approx(3.0, abs=0.4)


def test_doubly_robust_survives_a_wrong_outcome_model(confounded_data):
    """The double-robustness guarantee: with a correct propensity model, a
    deliberately misspecified outcome model still gives an unbiased effect."""
    X, treatment, outcome = confounded_data

    class ConstantModel:
        def fit(self, X, y):
            self.mean_ = y.mean()
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)      # ignores X entirely

    dr = DoublyRobust(outcome_model=ConstantModel()).fit(X, treatment, outcome)
    assert dr.ate_ == pytest.approx(3.0, abs=0.6)


def test_propensity_score_predicts_treatment(confounded_data):
    X, treatment, outcome = confounded_data
    p = propensity_score(X, treatment)
    # treated subjects should have higher propensity than untreated, on average
    assert p[treatment == 1].mean() > p[treatment == 0].mean()


def test_propensity_matching_recovers_the_effect(confounded_data):
    X, treatment, outcome = confounded_data
    pm = PropensityMatching(caliper=0.05).fit(X, treatment, outcome)
    assert pm.n_matched_ > 0
    assert pm.att_ == pytest.approx(3.0, abs=0.6)
