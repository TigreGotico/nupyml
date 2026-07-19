"""Split a document at TOPIC shifts by lexical cohesion (Hearst, 1997)."""
import re
from collections import Counter, defaultdict
import numpy as np


def _tokenize(text):
    return _WORD.findall(text.lower())


_WORD = re.compile(r"[A-Za-z']+")


def text_tiling(text, block_size=10, gap_step=5, smoothing=1):
    """Split a document at TOPIC shifts by lexical cohesion (Hearst, 1997).

    Within a topic the vocabulary is stable; at a topic boundary it changes. TextTiling
    reads that directly: slide two adjacent windows of words across the text and, at
    each gap, measure how similar their word distributions are (a cosine). Cohesion
    dips into a VALLEY wherever the vocabulary turns over, and the deepest valleys are
    the segment boundaries -- an unsupervised segmentation using nothing but word
    overlap. Returns the token offsets where topics change.
    """
    tokens = _tokenize(text)
    n = len(tokens)
    gaps = list(range(block_size, n - block_size, gap_step))
    scores = []
    for g in gaps:
        left = Counter(tokens[g - block_size:g])
        right = Counter(tokens[g:g + block_size])
        vocab = set(left) | set(right)
        a = np.array([left[w] for w in vocab]); b = np.array([right[w] for w in vocab])
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        scores.append(cos)
    scores = np.array(scores)
    for _ in range(smoothing):                            # smooth the cohesion curve
        scores = np.convolve(scores, [0.25, 0.5, 0.25], mode="same")
    # boundaries at valleys deeper than mean - std
    boundaries = []
    thresh = scores.mean() - scores.std()
    for i in range(1, len(scores) - 1):
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1] \
                and scores[i] < thresh:
            boundaries.append(gaps[i])
    return boundaries


__all__ = ["text_tiling"]
