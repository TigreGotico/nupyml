"""Probabilistic graphical models: factored joints, exact inference, structure.

The tests build networks with known conditional structure and check that
inference reproduces hand-computed posteriors, that fitting recovers the
generating tables, and that structure learning recovers the dependency graph the
data was generated from.
"""
import numpy as np
import pytest

from nupyml.pgm import (BayesianNetwork, DiscreteCPD, MarkovRandomField,
                        MarkovChain, variable_elimination, belief_propagation,
                        chow_liu, hill_climb_structure)
from nupyml.pgm._factor import Factor


# --- the factor algebra ---------------------------------------------------

def test_factor_multiply_aligns_shared_variables():
    a = Factor([0], [2], [0.5, 0.5])
    b = Factor([0, 1], [2, 2], [[0.9, 0.1], [0.2, 0.8]])
    prod = a.multiply(b)
    assert set(prod.variables) == {0, 1}
    # multiplying P(0) into P(1|0) gives the joint; it sums to 1 here
    assert prod.values.sum() == pytest.approx(1.0)


def test_factor_marginalize_sums_out():
    f = Factor([0, 1], [2, 2], [[0.1, 0.2], [0.3, 0.4]])
    m = f.marginalize([1])
    assert m.variables == (0,)
    assert np.allclose(m.values, [0.3, 0.7])


def test_factor_reduce_conditions_on_evidence():
    f = Factor([0, 1], [2, 3], np.arange(6).reshape(2, 3))
    r = f.reduce({1: 2})
    assert r.variables == (0,)
    assert np.allclose(r.values, [2, 5])          # column index 2 of each row


# --- Bayesian network -----------------------------------------------------

@pytest.fixture
def sprinkler_net():
    """Rain -> Sprinkler, {Rain, Sprinkler} -> Wet (0=no, 1=yes)."""
    bn = BayesianNetwork(edges=[(0, 1), (0, 2), (1, 2)])
    bn.add_cpd(DiscreteCPD(0, 2, [0.8, 0.2]))
    bn.add_cpd(DiscreteCPD(1, 2, np.array([[0.6, 0.99], [0.4, 0.01]]),
                           parents=[0], parent_cardinalities=[2]))
    pw = np.zeros((2, 2, 2))
    pw[1, 0, 0] = 0.0; pw[1, 1, 0] = 0.9; pw[1, 0, 1] = 0.8; pw[1, 1, 1] = 0.99
    pw[0] = 1 - pw[1]
    bn.add_cpd(DiscreteCPD(2, 2, pw, parents=[1, 0], parent_cardinalities=[2, 2]))
    return bn


def test_bayes_net_query_returns_a_normalized_marginal(sprinkler_net):
    q = sprinkler_net.query([2])
    assert q.variables == (2,)
    assert q.values.sum() == pytest.approx(1.0)


def test_evidence_updates_the_posterior(sprinkler_net):
    """Observing wet grass should raise the probability of rain above its prior --
    explaining-away in action."""
    prior = sprinkler_net.query([0]).values[1]         # P(Rain=yes) = 0.2
    posterior = sprinkler_net.query([0], evidence={2: 1}).values[1]
    assert prior == pytest.approx(0.2)
    assert posterior > prior


def test_bayes_net_recovers_its_cpds_from_samples(sprinkler_net):
    """Sample the net, refit, and the learned tables should match the true ones."""
    data = sprinkler_net.sample(4000, random_state=0)
    refit = BayesianNetwork(edges=[(0, 1), (0, 2), (1, 2)]).fit(
        data, columns=[0, 1, 2])
    assert refit.cpds_[0].factor.values[1] == pytest.approx(0.2, abs=0.03)
    # and the refit posterior matches the generating net's
    assert refit.query([0], evidence={2: 1}).values[1] == pytest.approx(
        sprinkler_net.query([0], evidence={2: 1}).values[1], abs=0.05)


def test_ancestral_sampling_respects_the_marginals(sprinkler_net):
    data = sprinkler_net.sample(5000, random_state=0)
    assert data[:, 0].mean() == pytest.approx(0.2, abs=0.03)    # P(Rain=yes)


def test_bayes_net_rejects_a_cyclic_graph():
    bn = BayesianNetwork(edges=[(0, 1), (1, 2), (2, 0)])
    bn.add_cpd(DiscreteCPD(0, 2, [0.5, 0.5]))
    with pytest.raises(ValueError, match="cycle"):
        bn._topological_order([0, 1, 2])


def test_bayes_net_log_likelihood_prefers_the_true_structure():
    """The generating structure should score higher log-likelihood than a wrong
    one on data it produced."""
    true = BayesianNetwork(edges=[(0, 1)])
    true.add_cpd(DiscreteCPD(0, 2, [0.5, 0.5]))
    true.add_cpd(DiscreteCPD(1, 2, np.array([[0.9, 0.1], [0.1, 0.9]]),
                             parents=[0], parent_cardinalities=[2]))
    data = true.sample(2000, random_state=0)

    right = BayesianNetwork(edges=[(0, 1)]).fit(data, columns=[0, 1])
    empty = BayesianNetwork(edges=[]).fit(data, columns=[0, 1])
    # the dependency 0->1 is real, so keeping it explains the data better
    assert right.log_likelihood(data) > empty.log_likelihood(data)


# --- inference ------------------------------------------------------------

def test_variable_elimination_matches_brute_force():
    """On a small net, elimination must equal the answer from the full joint."""
    bn = BayesianNetwork(edges=[(0, 1)])
    bn.add_cpd(DiscreteCPD(0, 2, [0.3, 0.7]))
    bn.add_cpd(DiscreteCPD(1, 2, np.array([[0.8, 0.4], [0.2, 0.6]]),
                           parents=[0], parent_cardinalities=[2]))
    # brute force: full joint P(0,1) = P(0)P(1|0), then marginalise 0
    joint = np.array([[0.3 * 0.8, 0.3 * 0.2], [0.7 * 0.4, 0.7 * 0.6]])
    expected = joint.sum(axis=0)                   # P(1)
    q = bn.query([1])
    assert np.allclose(q.values, expected)


def test_belief_propagation_agrees_with_elimination(sprinkler_net):
    factors = [cpd.to_factor() for cpd in sprinkler_net.cpds_.values()]
    bp = belief_propagation(factors, [0, 2])
    ve0 = variable_elimination(factors, 0)
    assert np.allclose(bp[0].values, ve0.values)


def test_elimination_order_does_not_change_the_answer():
    bn = BayesianNetwork(edges=[(0, 1), (1, 2)])
    bn.add_cpd(DiscreteCPD(0, 2, [0.5, 0.5]))
    bn.add_cpd(DiscreteCPD(1, 2, np.array([[0.7, 0.3], [0.3, 0.7]]),
                           parents=[0], parent_cardinalities=[2]))
    bn.add_cpd(DiscreteCPD(2, 2, np.array([[0.6, 0.2], [0.4, 0.8]]),
                           parents=[1], parent_cardinalities=[2]))
    factors = [cpd.to_factor() for cpd in bn.cpds_.values()]
    a = variable_elimination(factors, 2, elimination_order=[0, 1])
    b = variable_elimination(factors, 2, elimination_order=[1, 0])
    assert np.allclose(a.values, b.values)


# --- Markov chain and MRF -------------------------------------------------

def test_markov_chain_recovers_transitions():
    """Fit on a deterministic cycle; the transition matrix should be ~one-hot."""
    mc = MarkovChain(alpha=0.01).fit([0, 1, 2] * 100, n_states=3)
    assert mc.transition_[0, 1] > 0.95            # 0 is (almost) always -> 1
    assert mc.transition_[2, 0] > 0.95


def test_markov_chain_stationary_distribution():
    """A symmetric two-state chain has a uniform stationary distribution."""
    mc = MarkovChain()
    mc.n_states_ = 2
    mc.transition_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    mc.initial_ = np.array([1.0, 0.0])
    pi = mc.stationary_distribution()
    assert np.allclose(pi, [0.5, 0.5], atol=1e-6)
    # and it satisfies pi T = pi
    assert np.allclose(pi @ mc.transition_, pi, atol=1e-6)


def test_markov_chain_biased_stationary():
    """A chain that favours one state should concentrate its stationary mass
    there."""
    mc = MarkovChain()
    mc.n_states_ = 2
    mc.transition_ = np.array([[0.9, 0.1], [0.5, 0.5]])   # sticky in state 0
    mc.initial_ = np.array([0.5, 0.5])
    pi = mc.stationary_distribution()
    assert pi[0] > pi[1]


def test_mrf_partition_function_and_marginal():
    """A symmetric 'agree' potential gives a uniform marginal after normalising
    by Z."""
    f = Factor([0, 1], [2, 2], [[2.0, 1.0], [1.0, 2.0]])
    mrf = MarkovRandomField([f])
    assert mrf.partition_function() == pytest.approx(6.0)
    assert np.allclose(mrf.query([0]).values, [0.5, 0.5])


def test_mrf_coupling_makes_variables_agree():
    """With a strong agreement potential, conditioning one variable pulls the
    other toward the same value."""
    f = Factor([0, 1], [2, 2], [[5.0, 1.0], [1.0, 5.0]])
    mrf = MarkovRandomField([f])
    post = mrf.query([1], evidence={0: 0})
    assert post.values[0] > post.values[1]        # 1 likely matches 0


# --- structure learning ---------------------------------------------------

@pytest.fixture
def chain_data():
    """A dependency chain 0-1-2 (each copies its neighbour ~90%), with 3
    independent."""
    rng = np.random.RandomState(0)
    n = 3000
    x0 = rng.randint(0, 2, n)
    x1 = np.where(rng.uniform(size=n) < 0.9, x0, 1 - x0)
    x2 = np.where(rng.uniform(size=n) < 0.9, x1, 1 - x1)
    x3 = rng.randint(0, 2, n)
    return np.column_stack([x0, x1, x2, x3])


def test_chow_liu_recovers_the_dependency_chain(chain_data):
    """The strong dependencies 0-1 and 1-2 must be in the tree; 3 (independent)
    joins only as a weak leaf."""
    edges = chow_liu(chain_data)
    undirected = {frozenset(e) for e in edges}
    assert frozenset({0, 1}) in undirected
    assert frozenset({1, 2}) in undirected
    assert len(edges) == 3                         # a spanning tree of 4 nodes


def test_chow_liu_attaches_independent_variable_weakly(chain_data):
    """Variable 3 is independent of the rest, so its tree edge carries the least
    mutual information -- it is a leaf, connected to just one node."""
    edges = chow_liu(chain_data)
    degree = {}
    for i, j in edges:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    assert degree.get(3, 0) == 1                   # attached by a single edge


def test_hill_climb_leaves_independent_variable_disconnected(chain_data):
    """Unlike Chow-Liu (which must span), the score-based search should NOT
    connect the independent variable 3 at all -- an edge to it would not earn its
    BIC penalty."""
    edges = hill_climb_structure(chain_data, max_parents=2, max_iter=30)
    involved = {v for e in edges for v in e}
    assert 3 not in involved
    # and it should connect the real chain (0-1-2 in some orientation)
    undirected = {frozenset(e) for e in edges}
    assert frozenset({1, 2}) in undirected or frozenset({0, 1}) in undirected
