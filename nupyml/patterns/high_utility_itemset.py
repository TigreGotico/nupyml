"""High-utility itemset mining: find itemsets that are PROFITABLE, not just frequent."""
from itertools import combinations


def high_utility_itemsets(transactions, external_utility, min_utility):
    """Mine itemsets by total UTILITY, where frequency misses the point (Yao et al., 2004).

    Frequent-itemset mining counts occurrences, but "bread" sells far more often than
    "caviar" while earning less. High-utility mining scores an itemset by total VALUE --
    quantity times per-item profit, summed over the transactions containing it -- so it
    surfaces the profitable combinations a support threshold would drown out. Utility has
    no downward-closure (a superset can out-earn its subsets), so instead it prunes with
    the TRANSACTION-WEIGHTED UTILITY: an over-estimate that DOES close downward, letting
    whole branches be discarded before their exact utility is computed. ``transactions``
    map item -> quantity; ``external_utility`` maps item -> unit profit; returns itemsets
    whose total utility reaches ``min_utility``.
    """
    # transaction utility = sum of every item's utility in it
    tu = [sum(q * external_utility.get(it, 0) for it, q in t.items())
          for t in transactions]
    items = sorted({it for t in transactions for it in t})
    # TWU(item) = sum of TU over transactions containing it -- a closable upper bound
    twu = {it: sum(tu[i] for i, t in enumerate(transactions) if it in t)
           for it in items}
    promising = [it for it in items if twu[it] >= min_utility]

    def utility(itemset):
        total = 0
        for i, t in enumerate(transactions):
            if all(it in t for it in itemset):
                total += sum(t[it] * external_utility.get(it, 0) for it in itemset)
        return total

    results = []
    # only combinations drawn from TWU-promising items can be high-utility
    for k in range(1, len(promising) + 1):
        any_kept = False
        for combo in combinations(promising, k):
            # prune: an itemset's TWU is bounded by the min TWU of its items
            if min(twu[it] for it in combo) < min_utility:
                continue
            u = utility(combo)
            if u >= min_utility:
                results.append((combo, u)); any_kept = True
        if not any_kept and k > 1:
            break
    return sorted(results, key=lambda r: -r[1])


__all__ = ["high_utility_itemsets"]
