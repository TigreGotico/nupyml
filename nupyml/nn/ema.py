"""Exponential Moving Average of the weights -- a smoothed "shadow" model."""


class EMA:
    """Exponential Moving Average of the weights -- a smoothed "shadow" model.

    Keep a decayed running average of the parameters
    (``shadow = decay*shadow + (1-decay)*weights``). Evaluating with the shadow
    weights is a cheap, standard trick that noticeably improves stability and
    final accuracy (and it is what diffusion / GAN training almost always report).
    """

    def __init__(self, params, decay=0.999):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.data.copy() for p in self.params]

    def update(self):
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s *= d; s += (1 - d) * p.data

    def copy_to(self):
        for p, s in zip(self.params, self.shadow):
            p.data[...] = s


# --- schedules ------------------------------------------------------------


__all__ = ["EMA"]
