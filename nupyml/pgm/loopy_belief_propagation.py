"""Approximate marginals by passing messages until they stop changing."""
import numpy as np
from ._factor import Factor


def loopy_belief_propagation(factors, query, evidence=None, max_iter=50,
                             tol=1e-4, damping=0.0):
    """Approximate marginals by passing messages until they stop changing.

    Build a factor graph (variables on one side, factors on the other) and
    iterate: each variable tells each neighbouring factor the product of what its
    OTHER factors said; each factor tells each neighbouring variable the product
    of its potential with what its OTHER variables said, summed over everything
    else. On a tree this converges in one sweep to the exact answer; on a LOOPY
    graph it usually still converges, to a good approximation -- the basis of
    turbo/LDPC decoding. ``damping`` blends each new message with the old to help
    convergence oscillating cases.
    """
    evidence = evidence or {}
    factors = [f.reduce(evidence) if any(v in evidence for v in f.variables)
               else f.copy() for f in factors]
    factors = [f for f in factors if f.variables]        # drop fully-observed
    variables = sorted({v for f in factors for v in f.variables})
    card = {}
    for f in factors:
        card.update(dict(zip(f.variables, f.cardinalities)))

    # messages: var->factor and factor->var, initialised uniform
    m_vf = {(v, fi): np.ones(card[v]) for fi, f in enumerate(factors)
            for v in f.variables}
    m_fv = {(fi, v): np.ones(card[v]) for fi, f in enumerate(factors)
            for v in f.variables}

    def normalize(x):
        s = x.sum()
        return x / s if s > 0 else x

    for _ in range(max_iter):
        max_delta = 0.0
        # variable -> factor: product of incoming factor messages except target
        for (v, fi), _old in list(m_vf.items()):
            msg = np.ones(card[v])
            for fj, f in enumerate(factors):
                if v in f.variables and fj != fi:
                    msg = msg * m_fv[(fj, v)]
            m_vf[(v, fi)] = normalize(msg)
        # factor -> variable: multiply potential by other vars' messages, sum out
        for (fi, v), old in list(m_fv.items()):
            f = factors[fi]
            belief = f.copy()
            for u in f.variables:
                if u != v:
                    fac = Factor((u,), (card[u],), m_vf[(u, fi)])
                    belief = belief.multiply(fac)
            marg = belief.marginalize([u for u in f.variables if u != v])
            new = normalize(marg._align((v,)).ravel())
            if damping:
                new = normalize((1 - damping) * new + damping * old)
            max_delta = max(max_delta, np.abs(new - old).max())
            m_fv[(fi, v)] = new
        if max_delta < tol:
            break

    # belief at the query variable = product of all its incoming factor messages
    b = np.ones(card[query])
    for fi, f in enumerate(factors):
        if query in f.variables:
            b = b * m_fv[(fi, query)]
    return normalize(b)


__all__ = ["loopy_belief_propagation"]
