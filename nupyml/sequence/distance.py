"""Distances between sequences that may be misaligned.

Element-by-element comparison assumes the sequences line up. They rarely do: one
recording is slower than another, one string has a typo, one signal is shifted.
Every function here answers "how far apart, ALLOWING FOR realignment" -- and every
one is a dynamic program, filling a table where each cell is the best way to match
two prefixes.
"""
import numpy as np


def dtw_distance(a, b, window=None):
    """Dynamic time warping: the cost of the best NONLINEAR alignment.

    THE PROBLEM IT SOLVES
    ---------------------
    Two sequences of the same shape but different speed -- the same word said
    slowly and quickly, two gait cycles at different paces. Euclidean distance
    compares position i to position i and reports them as wildly different,
    because it insists they march in lockstep. That is the wrong answer: they are
    the same shape.

    DTW instead finds the best way to STRETCH one onto the other -- one element may
    match several, time may pause or race -- and returns the cost of that best
    alignment. It is the standard distance for time series precisely because it is
    invariant to local speed.

    THE DYNAMIC PROGRAM
    -------------------
    Each cell ``D[i,j]`` is the cheapest alignment of the first ``i`` of ``a`` with
    the first ``j`` of ``b``::

        D[i,j] = |a_i - b_j| + min(D[i-1,j],     # b pauses (a advances)
                                   D[i,j-1],     # a pauses (b advances)
                                   D[i-1,j-1])   # both advance together

    The three choices ARE the warping: pause left, pause right, or step together.
    An exponential number of alignments collapses to an ``O(nm)`` table because
    the best alignment of two prefixes only ever extends the best alignment of
    shorter prefixes.

    ``window`` (Sakoe-Chiba band) forbids matching elements more than ``window``
    apart. It cuts cost to ``O(n * window)`` AND improves quality -- an unbounded
    warp can align the start of one series to the end of the other, a
    "pathological warp" that is cheap on paper and meaningless in fact.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        # restrict j to a band around the diagonal when a window is set
        if window is not None:
            lo = max(1, i - window)
            hi = min(m, i + window)
        else:
            lo, hi = 1, m
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j],      # a advances, b waits
                                 D[i, j - 1],      # b advances, a waits
                                 D[i - 1, j - 1])  # both advance
    return D[n, m]


def dtw_path(a, b):
    """The warping path itself: which elements of ``a`` map to which of ``b``.

    Fill the same table, then BACKTRACK from the corner, at each step going
    whichever way was cheapest to arrive. The path is the alignment -- plot it and
    you see exactly where one series was stretched to fit the other.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # backtrack: from the end, always retreat toward the cheapest predecessor
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(path)), D[n, m]


def levenshtein(a, b):
    """Edit distance: fewest single-character insert/delete/substitute edits.

    The string analogue of DTW, and the same dynamic program with the same three
    moves -- insert, delete, substitute -- each costing 1. ``D[i,j]`` is the edit
    distance between the two prefixes; the answer is the far corner.

    OPTIMIZATION: only the previous row is ever read, so the full ``(n+1)*(m+1)``
    table collapses to two rows of length ``m+1`` -- ``O(m)`` memory instead of
    ``O(nm)``. The classic space-saving that makes edit distance practical on long
    strings.
    """
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = np.arange(m + 1)      # editing an empty prefix costs one per character
    for i in range(1, n + 1):
        curr = np.empty(m + 1)
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1,        # delete from a
                          curr[j - 1] + 1,    # insert into a
                          prev[j - 1] + cost)  # substitute (free if equal)
        prev = curr
    return int(prev[m])


def damerau_levenshtein(a, b):
    """Levenshtein plus TRANSPOSITION of two adjacent characters as one edit.

    "teh" -> "the" is one swap, which plain Levenshtein charges as two
    substitutions. Transposition is the commonest human typo, so this variant is
    the better string-similarity measure for anything typed. The recurrence gains
    one more case -- look back two positions and check for a swap.
    """
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # need two rows back for the transposition check, so keep the full table
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
            # the transposition case: the last two characters are swapped
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                D[i, j] = min(D[i, j], D[i - 2, j - 2] + 1)
    return int(D[n, m])


def hamming(a, b):
    """Positionwise mismatches. Requires equal length -- it does NOT realign.

    The degenerate case of edit distance with only substitutions allowed. Fast
    and meaningful ONLY when the sequences are already aligned (fixed-width codes,
    DNA reads of equal length). One insertion at the front makes every subsequent
    position mismatch, which is exactly the failure the other measures exist to
    avoid.
    """
    a, b = str(a), str(b)
    if len(a) != len(b):
        raise ValueError("Hamming distance needs equal-length sequences")
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def longest_common_subsequence(a, b):
    """Length of the longest subsequence common to both (not contiguous).

    The dynamic program behind ``diff``: matching characters extend the diagonal,
    mismatches take the better of dropping one character from either side. A
    SUBSEQUENCE, not a substring -- the shared characters need only keep their
    order, not their adjacency, which is what makes it a measure of similarity
    rather than of literal overlap.
    """
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                D[i, j] = D[i - 1, j - 1] + 1
            else:
                D[i, j] = max(D[i - 1, j], D[i, j - 1])
    return int(D[n, m])


def jaro(a, b):
    """Jaro similarity in [0, 1]: designed for short strings like names.

    Not an edit distance -- a similarity built from two counts: how many
    characters MATCH (appear in both, within a sliding window), and how many of
    those matches are TRANSPOSED. It rewards shared characters that are roughly in
    place, which suits typos and name variants, where edit distance's integer
    counts are coarse. 1 is identical, 0 shares nothing.
    """
    a, b = str(a), str(b)
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    # a character can match only within this window -- keeps "matches" meaningful
    reach = max(la, lb) // 2 - 1
    a_matched = [False] * la
    b_matched = [False] * lb

    matches = 0
    for i in range(la):
        lo = max(0, i - reach)
        hi = min(i + reach + 1, lb)
        for j in range(lo, hi):
            if not b_matched[j] and a[i] == b[j]:
                a_matched[i] = b_matched[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    # count transpositions among the matched characters, in order
    a_m = [a[i] for i in range(la) if a_matched[i]]
    b_m = [b[j] for j in range(lb) if b_matched[j]]
    transpositions = sum(c1 != c2 for c1, c2 in zip(a_m, b_m)) // 2

    return (matches / la + matches / lb
            + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a, b, prefix_weight=0.1, max_prefix=4):
    """Jaro, boosted for strings that share a PREFIX.

    Winkler's observation: matches at the START of a string matter more, because
    people mistype and abbreviate the ends far more than the beginnings. So a
    shared prefix (up to ``max_prefix`` characters) pulls the score up. It is the
    standard choice for record linkage and deduplicating names, where "Jonathan"
    and "Jon" should score high.
    """
    j = jaro(a, b)
    a, b = str(a), str(b)
    prefix = 0
    for c1, c2 in zip(a[:max_prefix], b[:max_prefix]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return j + prefix * prefix_weight * (1 - j)


def needleman_wunsch(a, b, match=1, mismatch=-1, gap=-1):
    """Global sequence alignment -- edit distance's twin from bioinformatics.

    The same dynamic program as Levenshtein, recast as a SCORE to MAXIMISE rather
    than a cost to minimise: matches earn, mismatches and gaps cost. That sign
    flip is the whole difference, and it is the natural framing for DNA and
    protein alignment, where the question is "how much do these share" rather than
    "how many edits apart". Returns the optimal alignment score.
    """
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    # a prefix aligned against nothing is all gaps
    D[:, 0] = np.arange(n + 1) * gap
    D[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            D[i, j] = max(D[i - 1, j - 1] + s,    # align the two characters
                          D[i - 1, j] + gap,      # gap in b
                          D[i, j - 1] + gap)      # gap in a
    return float(D[n, m])


__all__ = ["dtw_distance", "dtw_path", "levenshtein", "damerau_levenshtein",
           "hamming", "jaro", "jaro_winkler", "longest_common_subsequence",
           "needleman_wunsch"]
