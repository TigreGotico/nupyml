"""GSP: generalised sequential-pattern mining, Apriori-style over sequences."""
import math


def _is_subsequence(pattern, sequence):
    it = iter(sequence)
    return all(item in it for item in pattern)              # greedy ordered match


def gsp(sequences, min_support=0.5, max_length=None):
    """Mine frequent SUBSEQUENCES level by level (Srikant & Agrawal, 1996).

    GSP is Apriori generalised from sets to SEQUENCES. It finds frequent length-1
    patterns, then repeatedly JOINS frequent length-k patterns into length-(k+1)
    candidates -- two patterns combine when dropping the first item of one equals
    dropping the last item of the other -- and rescans the database to keep those above
    the support threshold. The downward-closure property still holds (a sequence cannot
    be frequent if a subsequence is not), so infrequent candidates are pruned before
    counting. Simpler to reason about than PrefixSpan, at the cost of repeated scans.
    ``sequences`` is a list of item lists; support is the fraction containing the pattern.
    """
    n = len(sequences)
    min_count = math.ceil(min_support * n)

    def support(pat):
        return sum(_is_subsequence(pat, s) for s in sequences)

    items = sorted({it for s in sequences for it in s})
    frequent = [(it,) for it in items if support((it,)) >= min_count]
    results = [(p, support(p)) for p in frequent]
    k = 1
    while frequent and (max_length is None or k < max_length):
        candidates = set()
        for a in frequent:
            for b in frequent:
                if a[1:] == b[:-1]:                         # join on the k-1 overlap
                    candidates.add(a + (b[-1],))
        frequent = []
        for c in candidates:
            sc = support(c)
            if sc >= min_count:
                frequent.append(c); results.append((c, sc))
        k += 1
    return sorted(results, key=lambda r: (-r[1], len(r[0])))


__all__ = ["gsp"]
