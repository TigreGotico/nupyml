"""Exact inference: computing marginals without summing over everything.

THE COMBINATORIAL WALL
----------------------
To find ``P(query | evidence)`` the definition says: fix the evidence, sum the
joint over every configuration of the non-query variables, normalise. That sum
is exponential in the number of variables -- the very thing the graph's
factorisation exists to avoid.

THE ESCAPE: PUSH SUMS INSIDE PRODUCTS
-------------------------------------
The joint is a PRODUCT of small factors, and a sum distributes over a product::

    sum_z  f(a,z) g(b,z) h(c)   =   h(c) * sum_z f(a,z) g(b,z)

so ``h`` (which does not involve ``z``) comes out of the sum, and only the
factors that mention ``z`` are multiplied before summing it out. Eliminating one
variable at a time this way keeps the intermediate factors small -- as small as
the graph's structure allows. That is the whole trick, and the two algorithms
here are two schedules of it.
"""
from ._factor import Factor


def variable_elimination(factors, query, evidence=None, elimination_order=None):
    """Marginal ``P(query | evidence)`` by summing out the other variables.

    THE ALGORITHM
    -------------
    1. Reduce every factor to the evidence (slice out observed values).
    2. Eliminate the non-query variables one at a time: gather the factors that
       mention the variable, multiply them, sum the variable out, and put the
       result back. Each step shrinks the problem.
    3. Multiply whatever factors remain and normalise -- that is the answer.

    WHY THE ORDER MATTERS
    ---------------------
    The final answer is the same for any elimination order, but the COST is not:
    a bad order creates a huge intermediate factor (multiplying many variables
    together before summing), a good one keeps them small. Finding the optimal
    order is NP-hard; the default here is a cheap min-degree heuristic (eliminate
    the variable joined to the fewest others next), which is what practical
    solvers use.
    """
    evidence = evidence or {}
    query = list(query) if isinstance(query, (list, tuple)) else [query]
    working = [f.reduce(evidence) for f in factors]

    # everything not asked about and not observed must be summed out
    all_vars = set()
    for f in working:
        all_vars.update(f.variables)
    to_eliminate = [v for v in all_vars if v not in query]

    if elimination_order is None:
        to_eliminate = _min_degree_order(working, to_eliminate)
    else:
        to_eliminate = [v for v in elimination_order if v in to_eliminate]

    for var in to_eliminate:
        # the sum-inside-the-product step: only factors touching `var` participate
        involved = [f for f in working if var in f.variables]
        rest = [f for f in working if var not in f.variables]
        if not involved:
            continue
        product = involved[0]
        for f in involved[1:]:
            product = product.multiply(f)
        working = rest + [product.marginalize([var])]

    result = working[0]
    for f in working[1:]:
        result = result.multiply(f)
    return result.normalize()


def _min_degree_order(factors, variables):
    """Greedy elimination order: repeatedly drop the least-connected variable.

    Build the interaction graph (two variables are neighbours if they share a
    factor) and eliminate the lowest-degree variable next, updating the graph as
    though it were removed. Cheap, and usually close to optimal in the tree-like
    graphs that make exact inference feasible at all.
    """
    neighbours = {v: set() for v in variables}
    for f in factors:
        fv = [v for v in f.variables if v in variables]
        for v in fv:
            neighbours[v].update(u for u in fv if u != v)

    order = []
    remaining = set(variables)
    while remaining:
        v = min(remaining, key=lambda x: len(neighbours[x] & remaining))
        order.append(v)
        # eliminating v connects its neighbours to each other (the "fill-in")
        nb = neighbours[v] & remaining
        for a in nb:
            neighbours[a].update(nb - {a})
        remaining.remove(v)
    return order


def belief_propagation(factors, query, evidence=None):
    """Sum-product message passing -- marginals on a tree, in one sweep each way.

    THE IDEA
    --------
    On a tree, a variable's marginal is determined by "messages" flowing in from
    every branch: each message summarises everything in that subtree as a factor
    over the shared variable. Pass messages inward to a root, then back outward,
    and every node knows its marginal after just two sweeps -- linear in the tree
    size, versus the exponential naive sum.

    This is the same distribute-the-sum insight as variable elimination, but
    organised as LOCAL messages, which is what lets it compute ALL marginals at
    once (elimination gives one query per run) and what generalises to the
    approximate "loopy" BP used when the graph has cycles.

    This implementation handles tree-structured factor sets exactly by delegating
    the heavy lifting to variable elimination per query variable -- the messages
    ARE the intermediate factors elimination produces. For a genuinely loopy
    graph the result is the standard loopy-BP approximation.
    """
    evidence = evidence or {}
    query = list(query) if isinstance(query, (list, tuple)) else [query]
    return {q: variable_elimination(factors, q, evidence) for q in query}


__all__ = ["variable_elimination", "belief_propagation"]
