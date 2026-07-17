"""Frequent-itemset mining and association-rule generation.

Transactions are given as a list of sets (or lists) of item labels. Support is
reported as a FRACTION of transactions, so thresholds are comparable across
datasets of different size.
"""
from itertools import combinations, chain

import numpy as np


def _support_counts(transactions):
    return [set(t) for t in transactions]


def apriori(transactions, min_support=0.1):
    """Level-wise frequent-itemset mining.

    THE ALGORITHM
    -------------
    Find frequent single items; from them generate candidate PAIRS, keep the
    frequent ones; from those generate candidate TRIPLES, and so on -- each level
    built only from the frequent sets of the level below.

    The apriori principle does the pruning: a candidate ``k``-itemset is generated
    only if ALL of its ``(k-1)``-subsets were frequent. Since an infrequent subset
    guarantees the superset is infrequent, this discards the vast majority of the
    ``2^d`` itemsets without ever counting them. The cost is one full pass over
    the transactions per level -- which is exactly what ``fpgrowth`` improves on.

    Returns ``{frozenset: support}`` for every frequent itemset.
    """
    tx = _support_counts(transactions)
    n = len(tx)
    min_count = min_support * n

    # level 1: the frequent single items
    counts = {}
    for t in tx:
        for item in t:
            counts[frozenset([item])] = counts.get(frozenset([item]), 0) + 1
    frequent = {k: v for k, v in counts.items() if v >= min_count}
    all_frequent = dict(frequent)

    k = 2
    while frequent:
        # candidates: unions of frequent (k-1)-sets whose every (k-1)-subset is
        # itself frequent (the apriori pruning)
        prev = list(frequent.keys())
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k and all(
                        frozenset(s) in frequent
                        for s in combinations(union, k - 1)):
                    candidates.add(union)

        counts = {c: 0 for c in candidates}
        for t in tx:                          # one rescan per level
            for c in candidates:
                if c <= t:
                    counts[c] += 1
        frequent = {c: v for c, v in counts.items() if v >= min_count}
        all_frequent.update(frequent)
        k += 1

    return {k: v / n for k, v in all_frequent.items()}


class _FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.link = None            # threads together all nodes of the same item


def fpgrowth(transactions, min_support=0.1):
    """FP-Growth: mine frequent itemsets WITHOUT generating candidates.

    THE TWO IDEAS
    -------------
    1. **Compress the data into an FP-tree.** Sort each transaction's items by
       frequency and insert it as a path; shared prefixes are stored once. A
       dense dataset collapses dramatically -- the tree, not the transactions, is
       what gets mined, and it fits where the raw data might not.
    2. **Mine by recursion on conditional trees, generating NO candidates.** For
       each item, gather the paths leading to it (its "conditional pattern base"),
       build a smaller tree from them, and recurse. Frequent itemsets fall out of
       the tree structure directly.

    Avoiding candidate generation is the whole win over Apriori: no exponential
    blow-up of sets to test, and only two passes over the data (one to count, one
    to build the tree) instead of one per level. On dense data it is far faster.

    Returns ``{frozenset: support}``.
    """
    tx = _support_counts(transactions)
    n = len(tx)
    min_count = min_support * n

    # pass 1: item frequencies, to drop infrequent items and order the rest
    item_counts = {}
    for t in tx:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
    frequent_items = {i: c for i, c in item_counts.items() if c >= min_count}
    if not frequent_items:
        return {}
    order = sorted(frequent_items, key=lambda i: (-frequent_items[i], str(i)))
    rank = {item: r for r, item in enumerate(order)}

    # pass 2: build the tree, inserting each transaction's frequent items in
    # frequency order so common prefixes coincide and compress
    root = _FPNode(None, None)
    headers = {item: None for item in frequent_items}     # item -> first node

    def insert(items, node):
        if not items:
            return
        first = items[0]
        child = node.children.get(first)
        if child is None:
            child = _FPNode(first, node)
            node.children[first] = child
            child.link = headers[first]       # push onto the item's node list
            headers[first] = child
        child.count += 1
        insert(items[1:], child)

    for t in tx:
        items = sorted([i for i in t if i in frequent_items], key=lambda i: rank[i])
        insert(items, root)

    result = {}

    def mine(headers, suffix):
        # process items least-frequent first, so each conditional tree is small
        for item in sorted(headers, key=lambda i: rank.get(i, 1e9), reverse=True):
            node = headers.get(item)
            if node is None:
                continue
            support = 0
            pattern_base = []       # the paths leading to this item, with counts
            while node is not None:
                path = []
                p = node.parent
                while p.item is not None:
                    path.append(p.item)
                    p = p.parent
                if path:
                    pattern_base.append((path[::-1], node.count))
                support += node.count
                node = node.link
            new_suffix = suffix | {item}
            if support >= min_count:
                result[frozenset(new_suffix)] = support / n
                # build the conditional tree from the pattern base and recurse --
                # no candidates are ever enumerated
                cond_counts = {}
                for path, cnt in pattern_base:
                    for it in path:
                        cond_counts[it] = cond_counts.get(it, 0) + cnt
                cond_frequent = {i for i, c in cond_counts.items() if c >= min_count}
                if cond_frequent:
                    cond_root = _FPNode(None, None)
                    cond_headers = {i: None for i in cond_frequent}

                    def cinsert(items, node):
                        if not items:
                            return
                        first = items[0]
                        child = node.children.get(first)
                        if child is None:
                            child = _FPNode(first, node)
                            node.children[first] = child
                            child.link = cond_headers[first]
                            cond_headers[first] = child
                        child.count += cnt_ref[0]
                        cinsert(items[1:], child)

                    for path, cnt in pattern_base:
                        cnt_ref = [cnt]
                        items = sorted([i for i in path if i in cond_frequent],
                                       key=lambda i: rank[i])
                        cinsert(items, cond_root)
                    mine(cond_headers, new_suffix)

    mine(headers, frozenset())
    return result


def eclat(transactions, min_support=0.1):
    """ECLAT: mine by intersecting TRANSACTION-ID sets.

    THE DUAL VIEW
    -------------
    Instead of "for each itemset, which transactions contain it", flip it: "for
    each item, WHICH transactions contain it" -- a tid-set. Then the support of
    ``{A, B}`` is simply the SIZE OF THE INTERSECTION of A's and B's tid-sets, and
    growing an itemset is intersecting one more tid-set.

    This vertical layout makes support a set operation rather than a data scan, so
    once the tid-sets are built the transactions are never touched again. It shines
    when tid-sets are small (sparse data) and the intersections stay cheap. Same
    apriori pruning, a completely different data representation -- worth seeing
    precisely because it reframes what "counting support" even means.

    Zaki (2000). Returns ``{frozenset: support}``.
    """
    tx = _support_counts(transactions)
    n = len(tx)
    min_count = min_support * n

    # the vertical database: each item -> the set of transaction ids holding it
    tid_sets = {}
    for tid, t in enumerate(tx):
        for item in t:
            tid_sets.setdefault(item, set()).add(tid)
    tid_sets = {frozenset([i]): s for i, s in tid_sets.items()
                if len(s) >= min_count}

    result = {}

    def extend(prefix_items, prefix_tids, remaining):
        for i, (item, tids) in enumerate(remaining):
            # support = size of the intersection, no data scan involved
            new_tids = prefix_tids & tids if prefix_tids else tids
            if len(new_tids) >= min_count:
                new_set = prefix_items | item
                result[frozenset(new_set)] = len(new_tids) / n
                # recurse with only the later items, to avoid generating a set twice
                extend(new_set, new_tids, remaining[i + 1:])

    items = list(tid_sets.items())
    extend(frozenset(), None, items)
    return result


def association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """Turn frequent itemsets into rules A => B, scored by confidence AND lift.

    For each frequent itemset, split it every way into antecedent A and
    consequent B and keep the rule when both thresholds pass. LIFT is enforced,
    not just confidence, for the reason in the module docstring: a
    high-confidence rule can be an artefact of a popular consequent, and only
    lift > 1 certifies a genuine association -- confidence with the base rate
    divided out.

    Returns a list of dicts with antecedent, consequent, support, confidence, lift.
    """
    rules = []
    supports = frequent_itemsets
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # every non-trivial way to split the set into A => B
        for r in range(1, len(items)):
            for antecedent in combinations(items, r):
                A = frozenset(antecedent)
                B = itemset - A
                a_support = supports.get(A)
                b_support = supports.get(B)
                if a_support is None or b_support is None:
                    continue
                confidence = support / a_support
                # lift divides confidence by B's base rate: > 1 means A genuinely
                # raises B's odds, not that B is merely common
                lift = confidence / b_support
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": A, "consequent": B, "support": support,
                        "confidence": confidence, "lift": lift})
    return sorted(rules, key=lambda r: r["lift"], reverse=True)


__all__ = ["apriori", "fpgrowth", "eclat", "association_rules"]
