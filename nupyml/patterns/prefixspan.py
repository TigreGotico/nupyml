"""PrefixSpan: sequential pattern mining by prefix-projected database growth."""
import math


def prefixspan(sequences, min_support=0.5, max_length=None):
    """Mine frequent SUBSEQUENCES by growing prefixes (Pei et al., 2001).

    Itemset mining (Apriori) ignores ORDER, but clickstreams, purchase histories and
    DNA are sequences where "A then B" differs from "B then A". PrefixSpan finds frequent
    ordered subsequences without ever generating candidates: it grows a pattern one item
    at a time and, for each extension, keeps only the PROJECTED database -- the suffixes
    of sequences that still contain the current prefix. Each projection is smaller than
    the last, so the search shrinks as patterns grow, and no infrequent candidate is ever
    built. ``sequences`` is a list of item lists; support is the fraction of sequences
    containing the pattern as a (not necessarily contiguous) subsequence.
    """
    n = len(sequences)
    min_count = math.ceil(min_support * n)
    results = []

    def project(prefix, projected):
        # projected: list of (seq_index, start_pos) -- where to resume scanning
        counts = {}
        for si, pos in projected:
            seen = set()
            for i in range(pos, len(sequences[si])):
                it = sequences[si][i]
                if it not in seen:                          # first occurrence per suffix
                    counts.setdefault(it, []).append((si, i + 1))
                    seen.add(it)
        for it, newproj in counts.items():
            support = len({si for si, _ in newproj})
            if support >= min_count:
                pattern = prefix + [it]
                results.append((tuple(pattern), support))
                if max_length is None or len(pattern) < max_length:
                    project(pattern, newproj)

    project([], [(si, 0) for si in range(n)])
    return sorted(results, key=lambda r: (-r[1], len(r[0])))


__all__ = ["prefixspan"]
