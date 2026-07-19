"""Decode a high-probability sequence without a full search (classic)."""


def beam_search(score_fn, start, vocab, max_len, beam_width=3, eos=None):
    """Decode a high-probability sequence without a full search (classic).

    Greedy decoding commits to the single best token each step and cannot recover
    from an early mistake; exhaustive search is exponential. Beam search keeps the
    ``beam_width`` best partial sequences at every step, extends each by every token,
    scores them (``score_fn(sequence) -> log-prob of the next tokens``), and prunes
    back to the top ``beam_width`` -- a bounded-width breadth-first search that finds
    much better sequences than greedy for a small constant cost. ``score_fn(seq)``
    returns a vector of next-token log-probabilities over ``vocab``.
    """
    beams = [(0.0, list(start))]
    finished = []
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if eos is not None and seq and seq[-1] == eos:
                finished.append((score, seq)); continue
            logits = score_fn(seq)
            for tok, lp in zip(vocab, logits):
                candidates.append((score + lp, seq + [tok]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    finished.extend(beams)
    return max(finished, key=lambda x: x[0])[1]


__all__ = ["beam_search"]
