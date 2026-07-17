"""Simulated annealing and tabu search: one point, and a reason to go uphill.

THE PROBLEM WITH HILL CLIMBING
------------------------------
Never accept a worse solution and you stop at the first local optimum. That is
the entire difficulty of black-box optimization in one sentence, and the two
methods here are two answers to it.

* **Simulated annealing** accepts a worse solution with a probability that falls
  over time.
* **Tabu search** always takes the best available move -- even an uphill one --
  and forbids returning to where it has just been.

The population methods elsewhere in this package dodge the problem by keeping
many candidates at once. These keep one, which makes them cheap in memory and
makes the escape mechanism explicit rather than emergent.
"""
import numpy as np

from ..utils import check_random_state


class SimulatedAnnealing:
    """Accept worse solutions early, stop accepting them later.

    THE METROPOLIS CRITERION
    ------------------------
    A proposed move that improves is always taken. One that worsens by ``dE`` is
    taken with probability::

        p = exp(-dE / T)

    Two things follow from that single expression, and they are the method:

    * **Bad moves are likelier than terrible ones.** ``p`` decays with ``dE``, so
      the search will hop over a small ridge readily and a mountain almost never.
      A random restart cannot make that distinction.
    * **The temperature tunes the whole trade-off.** At high ``T``, ``dE/T`` is
      near zero and ``p`` near 1: everything is accepted, and the search is a
      random walk that ignores the objective entirely. At ``T -> 0`` no worsening
      move is ever accepted and it is plain hill climbing. The cooling schedule
      walks continuously from one to the other.

    The physical metaphor is real, not decorative: this is the Boltzmann
    distribution, and the algorithm simulates cooling a metal slowly enough that
    its atoms find a low-energy crystal instead of freezing into a defect. Cool
    too fast and you get the defect -- a local optimum.

    TEMPERATURE HAS UNITS
    ---------------------
    ``T`` is compared against ``dE``, so it is measured in whatever the objective
    is measured in. A fixed default like ``T0 = 1.0`` is therefore not a neutral
    choice -- it is a claim that typical uphill moves cost about 1. If they
    actually cost 30, then ``exp(-30/1)`` is 1e-13, nothing is ever accepted, and
    the method silently degrades into the hill climbing it exists to avoid. It
    will not error; it will just quietly stop being simulated annealing.

    So ``T0="auto"`` (the default) measures it: take a short random walk, look at
    the uphill moves actually proposed, and set ``T0`` so that a healthy fraction
    of them would be accepted. The rule is the standard one --
    ``T0 = -mean(dE) / log(acceptance_ratio)`` inverts the Metropolis criterion
    for the target ratio. A fixed number is still accepted for when the scale is
    genuinely known.

    THE SCHEDULE
    ------------
    Geometric (``T *= alpha``) is the standard. Theory guarantees the global
    optimum under a logarithmic schedule, but that schedule is far too slow to
    use, so every practical run is a heuristic. Guarantees you cannot afford are
    not guarantees.

    Kirkpatrick, Gelatt & Vecchi (1983).
    """

    def __init__(self, func, x0, bounds=None, T0="auto", T_min=1e-6, alpha=0.99,
                 n_iterations=5000, step_size=0.1, initial_acceptance=0.8,
                 random_state=None):
        self.func = func
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self.T0 = T0
        self.initial_acceptance = initial_acceptance
        self.T_min = T_min
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.random_state = random_state

    def _propose(self, x, rng):
        step = rng.normal(0, self.step_size, size=len(x))
        if self.bounds is None:
            return x + step
        span = self.bounds[:, 1] - self.bounds[:, 0]
        return np.clip(x + step * span, self.bounds[:, 0], self.bounds[:, 1])

    def _calibrate_T0(self, rng, n_probe=50):
        """Set the temperature from the objective's actual scale.

        Walks at random from the start, collects the uphill costs, and inverts
        the Metropolis criterion for the target acceptance ratio. Costs a few
        dozen evaluations and removes the method's most common silent failure.
        """
        x = self.x0.copy()
        f = self.func(x)
        uphill = []
        for _ in range(n_probe):
            candidate = self._propose(x, rng)
            f_new = self.func(candidate)
            if f_new > f:
                uphill.append(f_new - f)
            x, f = candidate, f_new       # a free walk, accepting everything
        if not uphill:
            # a landscape with no uphill anywhere: any temperature will do,
            # since the Metropolis test is never reached
            return 1.0
        return float(-np.mean(uphill) / np.log(self.initial_acceptance))

    def run(self):
        rng = check_random_state(self.random_state)
        T = self._calibrate_T0(rng) if self.T0 == "auto" else float(self.T0)
        self.T0_ = T

        x = self.x0.copy()
        f = self.func(x)
        # the current point may wander uphill, so the best seen is tracked
        # separately -- otherwise annealing can END worse than it passed through
        best_x, best_f = x.copy(), f
        self.history_ = [best_f]
        self.temperatures_ = [T]
        n_accepted_worse = 0

        for _ in range(self.n_iterations):
            if T < self.T_min:
                break
            candidate = self._propose(x, rng)
            f_new = self.func(candidate)
            dE = f_new - f

            # improvements always accepted; worsenings on the Metropolis test
            if dE < 0 or rng.uniform() < np.exp(-dE / T):
                x, f = candidate, f_new
                if dE > 0:
                    n_accepted_worse += 1
                if f < best_f:
                    best_x, best_f = x.copy(), f

            T *= self.alpha
            self.history_.append(best_f)
            self.temperatures_.append(T)

        self.best_ = best_x
        self.best_fitness_ = float(best_f)
        self.n_accepted_worse_ = n_accepted_worse
        return self


class TabuSearch:
    """Always take the best available move; forbid going straight back.

    Where annealing escapes local optima with randomness, tabu search escapes
    them with MEMORY. It is deterministic given its neighbourhood: at a local
    optimum every neighbour is worse, so it takes the least-bad one and climbs
    out. Plain hill climbing would stop; the reason it could not do this is that
    it would immediately step back down again, and oscillate forever.

    The tabu list is what breaks the oscillation. Recently visited solutions are
    forbidden for ``tenure`` iterations, so the search is pushed on to somewhere
    new rather than bouncing between two points.

    ``tenure`` is the one real parameter, and both failures are worth knowing:
    too short and it cycles anyway (the list forgets before the loop closes);
    too long and it forbids so much that it is driven away from good regions it
    has barely examined.

    Glover (1986).
    """

    def __init__(self, func, x0, bounds, n_iterations=200, n_neighbors=20,
                 tenure=10, step_size=0.1, random_state=None):
        self.func = func
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_iterations = n_iterations
        self.n_neighbors = n_neighbors
        self.tenure = tenure
        self.step_size = step_size
        self.random_state = random_state

    def _key(self, x):
        """Continuous space has no repeated points, so 'visited' must mean
        'visited a region'. Rounding to a grid is what makes the tabu list
        meaningful here at all -- without it nothing would ever be forbidden."""
        span = self.bounds[:, 1] - self.bounds[:, 0]
        return tuple(np.round(x / (span * self.step_size)).astype(int))

    def run(self):
        rng = check_random_state(self.random_state)
        x = self.x0.copy()
        f = self.func(x)
        best_x, best_f = x.copy(), f
        tabu = {}
        self.history_ = [best_f]

        for it in range(self.n_iterations):
            span = self.bounds[:, 1] - self.bounds[:, 0]
            neighbors = np.clip(
                x + rng.normal(0, self.step_size, size=(self.n_neighbors, len(x))) * span,
                self.bounds[:, 0], self.bounds[:, 1])

            best_move, best_move_f = None, np.inf
            for cand in neighbors:
                key = self._key(cand)
                f_cand = self.func(cand)
                # the ASPIRATION criterion: a tabu move is allowed anyway if it
                # beats the best ever seen. The list exists to prevent cycling,
                # and a new global best is proof this move is not a cycle --
                # refusing it would be the memory overriding the objective.
                is_tabu = tabu.get(key, 0) > it and f_cand >= best_f
                if not is_tabu and f_cand < best_move_f:
                    best_move, best_move_f = cand, f_cand

            if best_move is None:   # every neighbour forbidden; wait one step out
                self.history_.append(best_f)
                continue

            # taken even when worse than the current point -- this is the climb
            x, f = best_move, best_move_f
            tabu[self._key(x)] = it + self.tenure
            if f < best_f:
                best_x, best_f = x.copy(), f
            self.history_.append(best_f)

        self.best_ = best_x
        self.best_fitness_ = float(best_f)
        return self


__all__ = ["SimulatedAnnealing", "TabuSearch"]
