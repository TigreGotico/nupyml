"""Lion: a sign-based optimizer, discovered by program SEARCH."""
import numpy as np


class Lion:
    """Lion: a sign-based optimizer, discovered by program SEARCH.

    THE SURPRISE
    ------------
    Found by symbolic program search over optimizer space, Lion updates using only
    the SIGN of an interpolated momentum::

        update = sign(beta1 * momentum + (1 - beta1) * gradient)
        w -= lr * (update + weight_decay * w)

    Because the step is ``sign(...)``, every parameter moves by the SAME magnitude
    -- only the direction varies. That makes it memory-light (one momentum buffer,
    no second moment like Adam's ``v``) and it matches or beats Adam on large
    models at lower memory cost. It is also a striking example of the field's
    recent turn: an optimizer nobody designed, found by a machine, that works.

    The sign means the learning rate must be much SMALLER than Adam's (the step is
    never scaled down by a small gradient), and weight decay matters more.

    Chen et al. (2023).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        b1, b2 = self.betas
        for p, m in zip(self.params, self._m):
            if p.grad is None:
                continue
            g = p.grad
            # step direction is the SIGN of an interpolation of momentum and grad
            update = np.sign(b1 * m + (1 - b1) * g)
            p.data -= self.lr * (update + self.weight_decay * p.data)
            # the momentum buffer uses the OTHER interpolation -- Lion decouples
            # the direction from the memory update
            m *= b2
            m += (1 - b2) * g


__all__ = ["Lion"]
