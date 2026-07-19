"""Stochastic Weight Averaging -- average the weights along the SGD trajectory."""


class SWA:
    """Stochastic Weight Averaging -- average the weights along the SGD trajectory.

    Late in training with a cyclic/high learning rate, SGD bounces around the rim
    of a wide flat basin rather than sitting at one point. Averaging those weights
    lands you nearer the basin's CENTRE -- a flatter solution that generalises
    better than any single iterate, for free. Call ``update_parameters()`` at each
    collection point; ``swap_in()`` copies the running average into the model.
    """

    def __init__(self, params):
        self.params = list(params)
        self.averaged = [p.data.copy() for p in self.params]
        self.n = 0

    def update_parameters(self):
        self.n += 1
        for avg, p in zip(self.averaged, self.params):
            avg += (p.data - avg) / self.n          # running mean of the iterates

    def swap_in(self):
        for p, avg in zip(self.params, self.averaged):
            p.data[...] = avg


__all__ = ["SWA"]
