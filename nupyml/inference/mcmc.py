"""MCMC: sampling from a distribution you can only evaluate up to a constant.

THE PROBLEM IT SOLVES
---------------------
Bayesian inference produces a posterior ``p(theta | data) = p(data | theta) p(theta) / Z``.
The numerator is easy; the normaliser ``Z`` is an integral over all of parameter
space, usually intractable. So you have a function proportional to the posterior
and no way to normalise it -- you cannot even confirm it sums to one, let alone
compute a mean or an interval from it.

MCMC's escape is to stop trying to normalise and instead DRAW SAMPLES. Build a
random walk whose long-run visiting frequency equals the target distribution;
then a histogram of where it has been IS the posterior, and any expectation is
just an average over the samples. The intractable ``Z`` cancels out entirely,
because every step compares two probabilities as a RATIO and ``Z`` appears in both.

That cancellation is the whole reason MCMC works, and it is worth seeing in each
method below: nowhere is ``Z`` ever needed.

THE METHODS, IN ORDER OF CLEVERNESS
-----------------------------------
* ``MetropolisHastings`` -- propose a random step; accept it with a probability
  that depends only on the ratio of target densities. Dead simple, works on
  anything, and mixes slowly in high dimensions because the proposals are blind.
* ``GibbsSampler`` -- update one coordinate at a time from its exact conditional.
  No rejection, no step size to tune -- when the conditionals are available.
* ``HamiltonianMC`` -- use the GRADIENT of the log-density to propose distant
  steps that are still accepted. Treats the sampler as a physical system rolling
  along the density's surface; dramatically better in high dimensions, which is
  why it (via NUTS) powers modern probabilistic programming.

THE HARD PART: DIAGNOSING CONVERGENCE
-------------------------------------
The chain is only correct in the LIMIT. Early samples reflect the arbitrary
starting point, not the target -- so they are discarded as BURN-IN. And
successive samples are correlated, so a thousand of them carry less information
than a thousand independent draws. The catch is that a chain can look perfectly
healthy while being nowhere near converged: it may be stuck in one mode, missing
others entirely, and you would not know from the samples alone. There is no
certificate of convergence, only diagnostics that can reveal failure but never
prove success.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class MetropolisHastings(BaseEstimator):
    """Random-walk Metropolis: propose, then accept on the density ratio.

    THE UPDATE
    ----------
    From the current ``theta``, propose ``theta' = theta + noise``. Accept it with
    probability::

        min(1,  p(theta') / p(theta))

    -- ALWAYS move to a more probable point, SOMETIMES to a less probable one, in
    proportion to how much less probable. That "sometimes" is essential: a walk
    that only ever climbs would get stuck on the first hill and never sample the
    rest of the distribution. Accepting downhill steps is what lets the chain
    explore the whole target rather than collapsing to its mode.

    Since only the RATIO enters, the normaliser cancels -- you pass an
    unnormalised log-density and never need ``Z``.

    THE ONE KNOB: STEP SIZE
    -----------------------
    Too small and the chain inches along, exploring at a crawl (high
    autocorrelation). Too large and almost every proposal is rejected, so it
    stands still. The folklore target is an acceptance rate around 0.234 for
    high-dimensional problems -- ``acceptance_rate_`` reports where you landed, and
    it is the first thing to check when a chain mixes badly.
    """

    def __init__(self, log_prob, step_size=0.5, random_state=None):
        self.log_prob = log_prob            # unnormalised log-density
        self.step_size = step_size
        self.random_state = random_state

    def sample(self, x0, n_samples, burn_in=1000):
        rng = check_random_state(self.random_state)
        x = np.atleast_1d(np.asarray(x0, float))
        current_lp = self.log_prob(x)
        samples = np.zeros((n_samples, len(x)))
        n_accept = 0

        total = n_samples + burn_in
        for i in range(total):
            # a symmetric gaussian proposal (this is the "random-walk" variant)
            proposal = x + self.step_size * rng.normal(size=len(x))
            prop_lp = self.log_prob(proposal)
            # accept on the LOG ratio -- Z cancels, so it never appears
            if np.log(rng.uniform()) < prop_lp - current_lp:
                x, current_lp = proposal, prop_lp
                if i >= burn_in:
                    n_accept += 1
            if i >= burn_in:
                samples[i - burn_in] = x

        self.acceptance_rate_ = n_accept / n_samples
        return samples


class GibbsSampler(BaseEstimator):
    """Update one coordinate at a time from its exact conditional distribution.

    THE IDEA
    --------
    Sampling a joint distribution directly is hard; sampling one variable given
    all the others is often easy and has a closed form. Gibbs sampling cycles
    through the coordinates, redrawing each from its full conditional
    ``p(theta_i | everything else)``. Every draw is exact and ALWAYS accepted --
    there is no rejection and no step size, because you are sampling the right
    conditional rather than proposing and testing.

    That is its appeal and its limit: it needs the conditionals in closed form,
    which you get for conjugate models and hierarchical Bayesian setups but not in
    general. When they are available it is the natural, tuning-free choice.

    THE CATCH
    ---------
    Gibbs moves only along the axes, one coordinate at a time. When variables are
    strongly correlated, the joint distribution is a narrow diagonal ridge, and
    axis-aligned steps can only shuffle slowly back and forth along it -- the
    chain mixes terribly exactly when the variables are most dependent. This is
    the same axis-alignment weakness as coordinate descent, and reparameterising
    to decorrelate the variables is the standard fix.

    ``conditional_samplers[i](x, rng)`` must return a draw of coordinate ``i``
    given the current full vector ``x``.
    """

    def __init__(self, conditional_samplers, random_state=None):
        self.conditional_samplers = conditional_samplers
        self.random_state = random_state

    def sample(self, x0, n_samples, burn_in=1000):
        rng = check_random_state(self.random_state)
        x = np.atleast_1d(np.asarray(x0, float)).copy()
        d = len(x)
        samples = np.zeros((n_samples + burn_in, d))
        for i in range(n_samples + burn_in):
            for j in range(d):
                # redraw coordinate j from its conditional, holding the rest --
                # note the update is IN PLACE, so later coordinates this sweep see
                # the freshly updated earlier ones
                x[j] = self.conditional_samplers[j](x, rng)
            samples[i] = x
        return samples[burn_in:]


class HamiltonianMC(BaseEstimator):
    """Use the gradient to propose distant, high-acceptance moves.

    THE INSIGHT
    -----------
    Random-walk Metropolis proposes blindly, so in high dimensions almost every
    large step lands somewhere improbable and is rejected -- it is reduced to
    tiny, slow steps. HMC uses the GRADIENT of the log-density to propose steps
    that follow the distribution's shape, so even long jumps stay probable and are
    accepted.

    THE PHYSICS
    -----------
    Picture the negative log-density as a landscape and the sampler as a puck
    given a random kick of momentum. It rolls along the surface under Hamiltonian
    dynamics -- trading potential for kinetic energy -- for several steps, then
    stops. Because energy is conserved, it can travel a long way across the
    distribution and end somewhere that is still probable. The trajectory is
    simulated by the LEAPFROG integrator, and a Metropolis step at the end
    corrects for the integrator's small energy drift, keeping the sampler exact.

    ``NUTS`` (the No-U-Turn Sampler) is HMC that tunes the trajectory length
    automatically -- it runs until the path starts doubling back -- and is what
    Stan and PyMC use. This is plain HMC with a fixed number of leapfrog steps;
    the leap to NUTS is about removing that one remaining knob.

    Duane et al. (1987); Neal (2011).
    """

    def __init__(self, log_prob, grad_log_prob, step_size=0.1, n_leapfrog=20,
                 random_state=None):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.random_state = random_state

    def sample(self, x0, n_samples, burn_in=1000):
        rng = check_random_state(self.random_state)
        x = np.atleast_1d(np.asarray(x0, float))
        samples = np.zeros((n_samples, len(x)))
        n_accept = 0

        for i in range(n_samples + burn_in):
            # a fresh random momentum kick each iteration
            p = rng.normal(size=len(x))
            current_p = p.copy()
            x_new = x.copy()

            # leapfrog: half-step momentum, full-step position, half-step momentum
            # -- the symmetric integrator that keeps the dynamics reversible, which
            # is what the Metropolis correction at the end relies on
            p = p + 0.5 * self.step_size * self.grad_log_prob(x_new)
            for _ in range(self.n_leapfrog):
                x_new = x_new + self.step_size * p
                p = p + self.step_size * self.grad_log_prob(x_new)
            p = p - 0.5 * self.step_size * self.grad_log_prob(x_new)

            # Metropolis on the TOTAL energy (potential + kinetic); this corrects
            # the leapfrog integrator's small energy error and keeps HMC exact
            current_energy = -self.log_prob(x) + 0.5 * current_p @ current_p
            proposed_energy = -self.log_prob(x_new) + 0.5 * p @ p
            if np.log(rng.uniform()) < current_energy - proposed_energy:
                x = x_new
                if i >= burn_in:
                    n_accept += 1
            if i >= burn_in:
                samples[i - burn_in] = x

        self.acceptance_rate_ = n_accept / n_samples
        return samples


def effective_sample_size(samples):
    """Roughly how many INDEPENDENT draws a correlated chain is worth.

    MCMC samples are autocorrelated, so ``n`` of them carry less information than
    ``n`` independent ones. ESS estimates the equivalent independent count by
    summing the autocorrelation until it goes negative. A chain of 10,000 with an
    ESS of 200 is a warning: it has explored far less than its length suggests.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    n = len(samples)
    ess = np.zeros(samples.shape[1])
    for d in range(samples.shape[1]):
        x = samples[:, d] - samples[:, d].mean()
        var = x @ x / n
        if var == 0:
            ess[d] = n
            continue
        # sum autocorrelations until the first negative one (Geyer's rule)
        rho_sum = 0.0
        for lag in range(1, min(n, 1000)):
            rho = (x[:-lag] @ x[lag:]) / (n * var)
            if rho < 0:
                break
            rho_sum += rho
        ess[d] = n / (1 + 2 * rho_sum)
    return ess


__all__ = ["MetropolisHastings", "GibbsSampler", "HamiltonianMC",
           "effective_sample_size"]
