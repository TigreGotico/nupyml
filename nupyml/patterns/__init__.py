"""Association rule mining: what goes with what, in a basket of items.

THE QUESTION
------------
Given millions of transactions -- shopping baskets, co-viewed pages, co-occurring
symptoms -- which sets of items appear together far more often than chance? The
famous (probably apocryphal) answer is "beer and diapers"; the real ones drive
store layout, recommendation, and cross-selling.

THREE MEASURES, AND WHY ALL THREE ARE NEEDED
--------------------------------------------
* **Support** -- how often an itemset appears. Filters out the merely rare, which
  are not actionable however interesting.
* **Confidence** -- P(B | A): given A in the basket, how often is B? This is the
  rule's apparent strength, and on its own it is MISLEADING: a rule can have high
  confidence purely because B is popular, not because A predicts it.
* **Lift** -- confidence divided by B's baseline frequency. Lift > 1 means A and B
  co-occur MORE than independence predicts -- a genuine association, with the
  popularity of B divided out. Lift is what confidence pretends to be.

THE COMPUTATIONAL WALL, AND THE WAY THROUGH
-------------------------------------------
With ``d`` items there are ``2^d`` possible itemsets -- unenumerable. Every
algorithm here escapes it via the APRIORI PRINCIPLE: if an itemset is infrequent,
every superset of it is infrequent too (adding items can only lower support). So
the moment ``{A, B}`` falls below the support threshold, every set containing it
is pruned unexamined. That downward-closure is what makes the search finite, and
the three algorithms differ only in how cleverly they exploit it:

* ``apriori`` -- generate candidates level by level, prune, rescan. Clear, and
  the origin of the principle.
* ``fpgrowth`` -- compress the data into a prefix tree and mine it WITHOUT ever
  generating candidates. Much faster, dramatically less rescanning.
* ``eclat`` -- represent each item by the SET of transactions it is in, and
  intersect those sets. Support becomes an intersection size.

Agrawal & Srikant (1994); Han, Pei & Yin (2000).
"""
from .rules import apriori, fpgrowth, eclat, association_rules

__all__ = ["apriori", "fpgrowth", "eclat", "association_rules"]
