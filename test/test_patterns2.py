"""M11: pattern mining v2 -- PrefixSpan, GSP, high-utility itemsets, frequent subgraphs.

PrefixSpan and GSP recover a planted ordered subsequence and agree with each other;
high-utility mining surfaces a rare but profitable item that a support threshold would
miss; frequent-subgraph mining finds a planted triangle motif.
"""
from nupyml.patterns import (prefixspan, gsp, high_utility_itemsets,
                            frequent_subgraphs)


SEQS = [["A", "X", "B", "Y", "C"], ["A", "B", "C"], ["Z", "A", "B", "Q", "C"],
        ["A", "B", "D"], ["A", "B", "C", "E"]]


def test_prefixspan_finds_planted_subsequence():
    patterns = dict(prefixspan(SEQS, min_support=0.6))
    assert ("A", "B", "C") in patterns
    assert patterns[("A", "B")] == 5           # A then B in every sequence


def test_gsp_matches_prefixspan():
    ps = dict(prefixspan(SEQS, min_support=0.6))
    gs = dict(gsp(SEQS, min_support=0.6))
    assert set(ps) == set(gs)                  # same frequent set, two algorithms
    assert all(ps[p] == gs[p] for p in ps)     # ... and identical supports


def test_high_utility_surfaces_rare_profitable_item():
    txns = [{"bread": 2, "milk": 1}, {"bread": 1, "caviar": 1},
            {"caviar": 2, "milk": 1}, {"bread": 3}, {"caviar": 1, "bread": 1}]
    util = {"bread": 1, "milk": 2, "caviar": 50}
    hui = dict(high_utility_itemsets(txns, util, min_utility=80))
    assert ("caviar",) in hui                  # rare (3/5) yet high utility
    assert ("bread",) not in hui               # frequent but low utility


def _triangle():
    return (["A", "B", "C"], [(0, 1, "e"), (1, 2, "e"), (0, 2, "e")])


def test_frequent_subgraphs_finds_triangle():
    graphs = [_triangle(), _triangle(), _triangle(),
              (["A", "B", "D"], [(0, 1, "e"), (1, 2, "e")]), _triangle()]
    fs = frequent_subgraphs(graphs, min_support=0.6, max_edges=3)
    triangles = [(p, s) for p, s in fs if len(p.edges) == 3]
    assert len(triangles) == 1
    p, s = triangles[0]
    assert s == 4                              # 4 of 5 graphs contain the triangle
    assert sorted(p.node_labels) == ["A", "B", "C"]
