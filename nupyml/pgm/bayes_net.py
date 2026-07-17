"""Bayesian networks: a joint distribution factored along a DAG.

Each variable depends directly only on its PARENTS in the graph, so the joint is
the product of one conditional table per variable::

    P(x1..xn) = product_i  P(x_i | parents(x_i))

That factorisation is the whole economy of the model: a variable with two binary
parents needs 4 numbers, not 2^n. Edges encode direct dependence (and, read
carefully, often causal direction).
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array
from ._factor import Factor
from .inference import variable_elimination


class DiscreteCPD:
    """A conditional probability table ``P(variable | parents)``.

    Stored as a factor over ``[variable] + parents`` whose slices along the
    variable axis sum to 1 for every parent configuration -- i.e. a proper
    conditional distribution. A CPD with no parents is just a prior.
    """

    def __init__(self, variable, cardinality, values, parents=None,
                 parent_cardinalities=None):
        self.variable = variable
        self.cardinality = cardinality
        self.parents = list(parents or [])
        self.parent_cardinalities = list(parent_cardinalities or [])
        variables = [variable] + self.parents
        cards = [cardinality] + self.parent_cardinalities
        self.factor = Factor(variables, cards, values)

    def to_factor(self):
        return self.factor.copy()


class BayesianNetwork(BaseEstimator):
    """A directed graphical model over discrete variables.

    Give it ``edges`` (parent -> child pairs) and either fit the CPDs from data
    or set them directly. Then ``query`` any variables given evidence, sample
    from the joint, or score a dataset's log-likelihood.
    """

    def __init__(self, edges=None):
        self.edges = list(edges or [])

    def _topological_order(self, variables):
        """Order variables so every node comes after its parents.

        Sampling and the chain-rule factorisation both need parents before
        children; a DAG always admits such an order, and its absence would mean
        the "graph" has a cycle and is not a valid Bayes net.
        """
        parents = {v: set() for v in variables}
        for p, c in self.edges:
            parents[c].add(p)
        order, remaining = [], set(variables)
        while remaining:
            ready = [v for v in remaining if not (parents[v] & remaining)]
            if not ready:
                raise ValueError("the edge set has a cycle; not a valid DAG")
            order.extend(sorted(ready))
            remaining -= set(ready)
        return order

    def fit(self, X, columns=None, cardinalities=None, alpha=1.0):
        """Learn every CPD from data by counting (with Laplace smoothing).

        The maximum-likelihood CPD is just normalised counts: of the rows where
        the parents took a given configuration, what fraction had each value of
        the child. ``alpha`` adds a pseudo-count so a parent configuration never
        seen in training still gets a valid (uniform-ish) distribution instead of
        divide-by-zero -- the same Laplace smoothing as naive Bayes.
        """
        X = np.asarray(X)
        n, d = X.shape
        self.columns_ = list(columns) if columns else list(range(d))
        col_idx = {c: i for i, c in enumerate(self.columns_)}
        if cardinalities is None:
            cardinalities = {c: int(X[:, col_idx[c]].max()) + 1
                             for c in self.columns_}
        self.cardinalities_ = cardinalities

        parents = {v: [] for v in self.columns_}
        for p, c in self.edges:
            parents[c].append(p)

        self.cpds_ = {}
        for v in self.columns_:
            pa = parents[v]
            card_v = cardinalities[v]
            pa_cards = [cardinalities[p] for p in pa]
            # count[value_of_v, *parent_values] with a Laplace prior
            counts = np.full([card_v] + pa_cards, alpha, dtype=float)
            vi = col_idx[v]
            for row in X:
                key = tuple([row[vi]] + [row[col_idx[p]] for p in pa])
                counts[key] += 1
            # normalise over the child axis for each parent configuration
            counts /= counts.sum(axis=0, keepdims=True)
            self.cpds_[v] = DiscreteCPD(v, card_v, counts, pa, pa_cards)
        return self

    def add_cpd(self, cpd):
        if not hasattr(self, "cpds_"):
            self.cpds_ = {}
            self.cardinalities_ = {}
            self.columns_ = []
        self.cpds_[cpd.variable] = cpd
        self.cardinalities_[cpd.variable] = cpd.cardinality
        if cpd.variable not in self.columns_:
            self.columns_.append(cpd.variable)
        return self

    def query(self, variables, evidence=None):
        """P(variables | evidence) by exact variable elimination.

        The network's CPDs ARE the factors; inference multiplies and sums them.
        Returns a normalised ``Factor`` over the query variables.
        """
        factors = [cpd.to_factor() for cpd in self.cpds_.values()]
        return variable_elimination(factors, variables, evidence)

    def sample(self, n_samples=1, random_state=None):
        """Ancestral sampling: draw each variable given its already-drawn parents.

        Walk the topological order so every parent is set before its child, and
        draw the child from its CPD column for that parent configuration. This
        generates exact samples from the joint in one pass -- no rejection, no
        MCMC -- which is the practical payoff of the DAG structure.
        """
        from ..utils import check_random_state
        rng = check_random_state(random_state)
        order = self._topological_order(self.columns_)
        col_idx = {c: i for i, c in enumerate(self.columns_)}
        out = np.zeros((n_samples, len(self.columns_)), dtype=int)
        for s in range(n_samples):
            assign = {}
            for v in order:
                cpd = self.cpds_[v]
                # build ONE index tuple: keep the v-axis, fix every parent axis to
                # its sampled value (slicing them one at a time would shift the
                # remaining axis indices out from under us)
                idx = [slice(None)] * len(cpd.factor.variables)
                for p in cpd.parents:
                    idx[cpd.factor.variables.index(p)] = assign[p]
                probs = np.asarray(cpd.factor.values[tuple(idx)]).ravel()
                assign[v] = rng.choice(cpd.cardinality, p=probs / probs.sum())
                out[s, col_idx[v]] = assign[v]
        return out

    def log_likelihood(self, X):
        """Sum of log P(row) over the data -- how well the network explains it.

        Each row's probability is the product of its CPD entries (log-sum over
        variables), so this is the score structure learning maximises.
        """
        X = np.asarray(X)
        col_idx = {c: i for i, c in enumerate(self.columns_)}
        total = 0.0
        for row in X:
            for v in self.columns_:
                cpd = self.cpds_[v]
                key = tuple([row[col_idx[v]]]
                            + [row[col_idx[p]] for p in cpd.parents])
                total += np.log(max(cpd.factor.values[key], 1e-300))
        return float(total)


__all__ = ["BayesianNetwork", "DiscreteCPD"]
