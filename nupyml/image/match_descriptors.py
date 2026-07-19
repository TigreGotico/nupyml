"""Match two sets of binary descriptors by Hamming distance + ratio test."""
import numpy as np


def match_descriptors(desc1, desc2, max_ratio=0.8):
    """Match two sets of binary descriptors by Hamming distance + ratio test."""
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), int)
    # hamming distance matrix via XOR bit counts
    D = (desc1[:, None, :] != desc2[None, :, :]).sum(axis=2)
    matches = []
    for i in range(len(desc1)):
        order = np.argsort(D[i])
        best, second = D[i, order[0]], D[i, order[1]] if len(order) > 1 else 1e9
        if best < max_ratio * (second + 1e-9):           # Lowe's ratio test
            matches.append((i, int(order[0])))
    return np.array(matches) if matches else np.empty((0, 2), int)


__all__ = ["match_descriptors"]
