"""Undirected models: Markov random fields and Markov chains.

Where a Bayesian network's edges are directed (and often causal), these encode
SYMMETRIC relationships -- "these variables are related", with no arrow.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from ._factor import Factor
from .inference import variable_elimination


class MarkovRandomField(BaseEstimator):
    """An undirected graphical model: a product of clique POTENTIALS.

    THE DIFFERENCE FROM A BAYES NET
    -------------------------------
    A Markov random field factorises the joint over an UNDIRECTED graph::

        P(x) = (1/Z) * product of potentials over the cliques

    The potentials are non-negative but need NOT be conditional distributions,
    and there is no direction -- which is exactly right for symmetric structure:
    neighbouring pixels in an image, adjacent spins in a lattice, friends in a
    social graph, where "A causes B" is meaningless but "A and B tend to agree"
    is the whole model.

    THE PRICE: THE PARTITION FUNCTION Z
    -----------------------------------
    A Bayes net is automatically normalised (its CPDs are distributions), so its
    joint costs nothing to evaluate. An MRF is NOT: the normaliser ``Z`` is a sum
    over every configuration, and computing it is as hard as inference itself.
    This is the same intractable ``Z`` that haunts the RBM and energy-based models
    elsewhere -- and the reason MRF learning leans on approximate methods. Here
    ``Z`` is computed exactly by multiplying all potentials and summing, which is
    fine for the small graphs where exact inference is feasible.

    ``potentials`` is a list of ``Factor`` objects over the cliques.
    """

    def __init__(self, potentials=None):
        self.potentials_ = list(potentials or [])

    def add_potential(self, factor):
        self.potentials_.append(factor)
        return self

    def partition_function(self):
        """Z = sum over all configurations of the product of potentials.

        Multiply every potential into one big factor and sum it out entirely.
        Tractable only for small graphs -- which is the standing limitation of
        exact MRF work.
        """
        joint = self.potentials_[0].copy()
        for f in self.potentials_[1:]:
            joint = joint.multiply(f)
        return float(joint.values.sum())

    def query(self, variables, evidence=None):
        """P(variables | evidence). Elimination normalises, so Z cancels out --
        which is why conditional queries are easier than computing Z itself."""
        factors = [p.copy() for p in self.potentials_]
        return variable_elimination(factors, variables, evidence)


class MarkovChain(BaseEstimator):
    """A first-order Markov chain over discrete states.

    THE ASSUMPTION
    --------------
    The future depends on the present only, not the fuller past::

        P(x_t | x_{t-1}, x_{t-2}, ...) = P(x_t | x_{t-1})

    So the whole model is one transition matrix ``T[i, j] = P(next=j | now=i)``,
    estimated by counting how often ``i`` is followed by ``j``. It is the
    simplest sequential model and the backbone of PageRank, text generation,
    queueing, and the HMM's hidden layer.

    THE STATIONARY DISTRIBUTION
    ---------------------------
    Run the chain long enough and the state distribution settles to a fixed point
    ``pi`` with ``pi T = pi`` -- the left eigenvector of ``T`` for eigenvalue 1.
    It answers "where does the chain spend its time in the long run", independent
    of where it started (for an irreducible, aperiodic chain). That eigenvector
    IS PageRank's ranking when ``T`` is the web's link matrix.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing for unseen transitions

    def fit(self, sequences, n_states=None):
        """Count transitions across one or more state sequences."""
        if isinstance(sequences[0], (int, np.integer)):
            sequences = [sequences]          # a single sequence
        flat = np.concatenate([np.asarray(s) for s in sequences])
        self.n_states_ = int(n_states or flat.max() + 1)
        k = self.n_states_

        counts = np.full((k, k), self.alpha, dtype=float)
        for seq in sequences:
            seq = np.asarray(seq)
            for i in range(len(seq) - 1):
                counts[seq[i], seq[i + 1]] += 1
        # each row is P(next | current), so normalise per row
        self.transition_ = counts / counts.sum(axis=1, keepdims=True)
        # initial-state distribution from the sequence starts
        init = np.full(k, self.alpha)
        for seq in sequences:
            init[np.asarray(seq)[0]] += 1
        self.initial_ = init / init.sum()
        return self

    def stationary_distribution(self):
        """The long-run state frequencies: the eigenvector of T' for eigenvalue 1.

        Solve ``pi T = pi`` as a left-eigenvector problem, take the one whose
        eigenvalue is 1, and normalise it to a distribution. This is the fixed
        point the chain converges to regardless of its start."""
        vals, vecs = np.linalg.eig(self.transition_.T)
        idx = np.argmin(np.abs(vals - 1.0))         # eigenvalue closest to 1
        pi = np.real(vecs[:, idx])
        pi = np.abs(pi)                             # sign is arbitrary; take |.|
        return pi / pi.sum()

    def sample(self, length, random_state=None):
        """Generate a state sequence by walking the chain."""
        rng = check_random_state(random_state)
        seq = [rng.choice(self.n_states_, p=self.initial_)]
        for _ in range(length - 1):
            seq.append(rng.choice(self.n_states_, p=self.transition_[seq[-1]]))
        return np.array(seq)

    def log_likelihood(self, sequence):
        seq = np.asarray(sequence)
        ll = np.log(max(self.initial_[seq[0]], 1e-300))
        for i in range(len(seq) - 1):
            ll += np.log(max(self.transition_[seq[i], seq[i + 1]], 1e-300))
        return float(ll)


__all__ = ["MarkovRandomField", "MarkovChain"]
