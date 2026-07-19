"""Wrap any optimizer: let it explore, then pull back toward a slow average."""


class Lookahead:
    """Wrap any optimizer: let it explore, then pull back toward a slow average.

    THE IDEA
    --------
    Lookahead is a META-optimizer. It keeps a "fast" inner optimizer (Adam, SGD,
    anything) and a "slow" copy of the weights. The fast optimizer takes ``k``
    normal steps, exploring ahead; then the slow weights move a FRACTION of the
    way toward where the fast ones landed, and the fast weights are reset to the
    slow ones. Explore ``k`` steps, commit a little, repeat.

    The slow average damps the oscillations that fast optimizers make in sharp or
    noisy loss landscapes, giving more stable convergence and less sensitivity to
    the inner optimizer's hyperparameters -- often better final accuracy for
    almost no extra cost. It composes with any optimizer, which is the elegant
    part: it improves the one you already use rather than replacing it.

    Zhang et al. (2019).
    """

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step = 0
        # the slow weights: a snapshot the fast optimizer is periodically pulled
        # back toward
        self._slow = [p.data.copy() for p in self.base.params]

    def zero_grad(self):
        self.base.zero_grad()

    def step(self):
        self.base.step()                        # a fast (inner) step
        self._step += 1
        if self._step % self.k == 0:
            # every k steps, move the slow weights partway to the fast ones and
            # reset the fast ones there -- the "look ahead, then commit" move
            for p, slow in zip(self.base.params, self._slow):
                slow += self.alpha * (p.data - slow)
                p.data[...] = slow


__all__ = ["Lookahead"]
