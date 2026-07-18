"""I10: PGM v2 -- hidden semi-Markov model, loopy BP, Gaussian Bayes net.

The HSMM must recover an explicit-duration segmentation a geometric-duration HMM
could not; loopy BP must equal exact inference on a tree and stay sane on a loop;
the Gaussian Bayes net must recover its linear coefficients and answer
conditional queries with the exact Gaussian formula.
"""
import numpy as np
import pytest

from nupyml.hmm import HiddenSemiMarkovModel
from nupyml.pgm import (loopy_belief_propagation, GaussianBayesianNetwork,
                        Factor, variable_elimination)
from nupyml.metrics import adjusted_rand_score


def _true_hsmm():
    m = HiddenSemiMarkovModel(n_states=2, max_duration=15, random_state=0)
    m.means_ = np.array([[0., 0.], [6., 6.]])
    m.covars_ = np.array([[.3, .3], [.3, .3]])
    m.startprob_ = np.array([1., 0.])
    m.transmat_ = np.array([[0., 1.], [1., 0.]])
    dur = np.zeros((2, 15)); dur[0, 9] = 1; dur[1, 7] = 1   # durations 10 and 8
    m.durations_ = dur
    return m


def test_hsmm_decodes_explicit_durations():
    true = _true_hsmm()
    X, states = true.sample(120, random_state=1)
    decoded = true.decode(X)
    assert (decoded == states).mean() > 0.95            # segmentation recovered


def test_hsmm_fit_recovers_segmentation_and_nongeometric_durations():
    true = _true_hsmm()
    X, states = true.sample(160, random_state=2)
    m = HiddenSemiMarkovModel(n_states=2, max_duration=15, random_state=0).fit(X)
    assert adjusted_rand_score(states, m.decode(X)) > 0.9
    # learned durations peak well away from 1 -- a geometric HMM cannot do this
    mean_dur = [(m.durations_[k] * np.arange(1, 16)).sum() for k in range(2)]
    assert min(mean_dur) > 4.0


def test_loopy_bp_matches_exact_inference_on_a_tree():
    fA = Factor(('A',), (2,), [0.6, 0.4])
    fAB = Factor(('A', 'B'), (2, 2), [[0.7, 0.3], [0.2, 0.8]])
    fBC = Factor(('B', 'C'), (2, 2), [[0.9, 0.1], [0.4, 0.6]])
    factors = [fA, fAB, fBC]
    lbp = loopy_belief_propagation(factors, 'C')
    ve = variable_elimination(factors, 'C')
    exact = ve.values / ve.values.sum()
    assert np.allclose(lbp, exact, atol=1e-3)           # exact on a tree


def test_loopy_bp_converges_on_a_loop():
    # symmetric attractive triangle -> uniform marginals by symmetry
    f1 = Factor(('A', 'B'), (2, 2), [[2, 1], [1, 2]])
    f2 = Factor(('B', 'C'), (2, 2), [[2, 1], [1, 2]])
    f3 = Factor(('A', 'C'), (2, 2), [[2, 1], [1, 2]])
    belief = loopy_belief_propagation([f1, f2, f3], 'A')
    assert np.allclose(belief, [0.5, 0.5], atol=1e-2)


def test_loopy_bp_responds_to_evidence():
    fAB = Factor(('A', 'B'), (2, 2), [[0.9, 0.1], [0.1, 0.9]])  # A and B agree
    belief = loopy_belief_propagation([fAB], 'B', evidence={'A': 0})
    assert belief[0] > belief[1]                         # A=0 pushes B toward 0


def test_gaussian_bayes_net_recovers_coefficients_and_conditions():
    rng = np.random.RandomState(0)
    X = rng.randn(500)
    Y = 2 * X + 1 + 0.1 * rng.randn(500)
    Z = -Y + 0.1 * rng.randn(500)
    gbn = GaussianBayesianNetwork({'X': [], 'Y': ['X'], 'Z': ['Y']}).fit(
        {'X': X, 'Y': Y, 'Z': Z})
    assert abs(gbn.coef_['Y']['X'] - 2.0) < 0.1
    assert abs(gbn.bias_['Y'] - 1.0) < 0.1
    # exact conditional query: X=1 -> Y ~ 3, Z ~ -3
    means, cov = gbn.condition({'X': 1.0})
    assert abs(means['Y'] - 3.0) < 0.1
    assert abs(means['Z'] - (-3.0)) < 0.1
