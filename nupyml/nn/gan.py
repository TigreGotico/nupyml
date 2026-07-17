"""Generative adversarial networks: learning a distribution by being judged.

THE IDEA
--------
Two networks with opposing jobs:

* the **generator** turns noise into fakes, and wants them believed;
* the **discriminator** looks at a sample and says real or fake.

Train them against each other. As the discriminator gets better at spotting
fakes, the generator is forced to make better ones, and vice versa. At the
theoretical equilibrium the generator's distribution equals the data's, and the
discriminator can do no better than a coin flip.

WHAT MAKES THIS DIFFERENT FROM A VAE
------------------------------------
A VAE maximises a likelihood, so it must define one -- which means committing to
a pixel-wise output distribution, which is why its samples are famously blurry:
squared error's optimal answer under uncertainty is the AVERAGE of the
possibilities, and the average of two plausible faces is a blurred face.

A GAN never writes down a likelihood. It only needs a critic to be fooled, and
"fooled" is not an average -- it is a judgement. That is why GANs produce sharp
samples, and also why they cannot tell you the probability of anything, cannot
score a held-out sample, and are hard to evaluate at all.

    VAE: tractable likelihood, blurry samples, stable training
    GAN: no likelihood, sharp samples, unstable training

WHY IT IS UNSTABLE
------------------
This is not one loss being minimised -- it is a two-player game, seeking a saddle
point rather than a minimum. Nothing decreases monotonically, so the loss curve
tells you almost nothing. Three specific failures:

*Vanishing gradient.* If the discriminator wins decisively it outputs ~0 for
every fake, its sigmoid saturates, and the generator's gradient goes to zero. The
better the discriminator, the less the generator learns -- exactly backwards.
``NonSaturatingGANLoss`` is the standard patch.

*Mode collapse.* The generator finds one output that reliably fools the
discriminator and emits only that. It has "won" while modelling almost none of
the data, and nothing in the objective punishes the missing variety.

*No progress signal.* Since neither loss measures distribution distance, you
cannot tell training apart from failure by watching them. This is precisely what
``WGAN`` fixes: its critic loss DOES estimate a distance, so it finally
correlates with sample quality.

A WARNING ABOUT BLAMING THE GAME
--------------------------------
The three failures above are real, and their reputation makes them the first
suspect whenever samples look wrong. That reputation is also a trap. A generator
with a Tanh output cannot emit anything outside [-1, 1]; point it at data centred
on 3 and the samples pile up against the ceiling -- which looks exactly like mode
collapse and is nothing of the sort. Check that the architecture CAN represent
the data before concluding that the adversarial game would not converge.
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Sequential
from .layers import Linear, LeakyReLU, Sigmoid, Tanh, BatchNorm1d
from .losses import BCEWithLogitsLoss


def _stack(sizes, rng, activation, final=None, batchnorm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng=rng))
        if i < len(sizes) - 2:
            if batchnorm:
                layers.append(BatchNorm1d(sizes[i + 1]))
            layers.append(activation())
    if final is not None:
        layers.append(final())
    return Sequential(*layers)


class Generator(Module):
    """Noise in, data out.

    The latent vector has no assigned meaning -- unlike a VAE's, it is never
    inferred from anything. It is simply a source of variety, and the generator
    learns whatever mapping from it fools the critic.

    BatchNorm is on by default in the generator: it is one of the few reliable
    stabilisers in the DCGAN recipe, because it keeps activations in range while
    the target (the discriminator) keeps moving underneath.

    ``output_activation`` defaults to None -- a linear output, which can reach any
    value. Pass ``Tanh`` when the data really is scaled to [-1, 1], as images
    usually are; it is the DCGAN convention and it helps. But it is a CEILING: a
    Tanh generator cannot emit 3.0 no matter how long it trains, and the failure
    is silent. The samples simply pile up against the boundary, which looks
    indistinguishable from mode collapse. Match the output activation to the range
    of the data before blaming the adversarial game.
    """

    def __init__(self, latent_dim, n_features, hidden=(64, 64), rng=None,
                 output_activation=None, batchnorm=True):
        super().__init__()
        rng = check_random_state(rng)
        self.latent_dim = latent_dim
        self.net = _stack([latent_dim, *hidden, n_features], rng, LeakyReLU,
                          final=output_activation, batchnorm=batchnorm)
        self._rng = rng

    def sample_latent(self, n):
        return Tensor(self._rng.normal(size=(n, self.latent_dim)))

    def forward(self, z):
        return self.net(Tensor._wrap(z))

    def sample(self, n, rng=None):
        was_training = self.training
        self.eval()
        rng = check_random_state(rng if rng is not None else self._rng)
        z = Tensor(rng.normal(size=(n, self.latent_dim)))
        out = self.forward(z).data
        self.train(was_training)
        return out


class Discriminator(Module):
    """Data in, one logit out: high means real.

    Emits a LOGIT, not a probability, so the loss can use the stable
    ``BCEWithLogitsLoss`` form -- and so the same class can serve as a WGAN
    critic, where the output is an unbounded score rather than a probability.

    LeakyReLU rather than ReLU is deliberate here: a dead ReLU in the
    discriminator kills the gradient the generator depends on, and the generator
    has no other teacher.
    """

    def __init__(self, n_features, hidden=(64, 64), rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.net = _stack([n_features, *hidden, 1], rng, LeakyReLU,
                          batchnorm=False)

    def forward(self, x):
        return self.net(Tensor._wrap(x)).reshape(-1)


class GANLoss(Module):
    """The original minimax objective.

    Discriminator: maximise ``log D(real) + log(1 - D(fake))``.
    Generator: MINIMISE ``log(1 - D(fake))``.

    Written as stated, the generator's loss is the problem this class exists to
    demonstrate. Early in training the discriminator easily rejects fakes, so
    ``D(fake) ~ 0``; but ``log(1 - D)`` is FLATTEST exactly there, so the
    generator gets almost no gradient precisely when it needs one most. Training
    stalls at the start, which is the worst possible time.

    Use ``NonSaturatingGANLoss`` in practice. This is here because seeing why the
    obvious formulation fails is the point.
    """

    def __init__(self):
        super().__init__()
        self._bce = BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits, fake_logits):
        real = self._bce(real_logits, np.ones(real_logits.shape))
        fake = self._bce(fake_logits, np.zeros(fake_logits.shape))
        return real + fake

    def generator_loss(self, fake_logits):
        # minimise log(1 - D(fake)) -- the saturating form
        return -self._bce(fake_logits, np.zeros(fake_logits.shape))


class NonSaturatingGANLoss(GANLoss):
    """The fix everyone actually uses: flip the generator's objective.

    Rather than MINIMISING ``log(1 - D(fake))``, MAXIMISE ``log D(fake)``::

        generator_loss = BCE(fake_logits, target=1)

    The generator now pretends its fakes are real and optimises for that. The two
    objectives share a fixed point -- the game is unchanged -- but the gradient
    landscape is not: this one is STEEPEST when ``D(fake) ~ 0``, exactly where the
    saturating form is flat. Same equilibrium, usable gradient.

    Proposed in the original GAN paper as a practical note, and it is what made
    the method work at all.
    """

    def generator_loss(self, fake_logits):
        return self._bce(fake_logits, np.ones(fake_logits.shape))


class WGANLoss(Module):
    """Wasserstein GAN: replace the classifier with a distance estimate.

    THE INSIGHT
    -----------
    A standard GAN implicitly minimises Jensen-Shannon divergence. When the real
    and generated distributions do not overlap -- which, for data on a
    low-dimensional manifold, is essentially always at the start -- JS is a
    CONSTANT (log 2). A constant has zero gradient, so the generator is told
    nothing about which way to move. That is the deep reason GANs are unstable,
    not a tuning problem.

    The Wasserstein (earth-mover) distance measures how far mass must be moved,
    so it is finite and has a useful gradient even for disjoint distributions. It
    always knows which direction is closer.

    Via Kantorovich-Rubinstein duality it becomes::

        W = sup over 1-Lipschitz f of  E[f(real)] - E[f(fake)]

    so the critic maximises that gap, the generator minimises it, and the critic's
    loss is now a real estimate of the distance -- which is why WGAN's loss curve
    actually tracks sample quality, unlike a standard GAN's.

    THE LIPSCHITZ CONSTRAINT
    ------------------------
    The duality only holds if the critic is 1-Lipschitz. Original WGAN clipped
    weights, which works but cripples capacity. WGAN-GP penalises the gradient
    norm instead -- see ``nupyml.nn.losses.gradient_penalty``.

    The critic is also trained several steps per generator step (``n_critic``),
    because the duality wants a critic near its optimum. In a standard GAN a
    too-good discriminator is fatal; here it is the goal. That inversion is the
    clearest sign the two objectives are genuinely different.

    Arjovsky, Chintala & Bottou (2017); Gulrajani et al. (2017).
    """

    def discriminator_loss(self, real_scores, fake_scores):
        # the critic maximises the gap, so minimise its negation
        return fake_scores.mean() - real_scores.mean()

    def generator_loss(self, fake_scores):
        return -fake_scores.mean()


class GAN(Module):
    """A generator and a discriminator, plus the alternating training step.

    The training loop is the interesting part, and ``step`` shows why:

    1. Update the discriminator on a batch of real and a batch of fake. The fakes
       are DETACHED -- the generator must not be updated by the discriminator's
       loss, since it is trying to make that loss worse, not better. Forgetting
       this detach is the classic GAN bug and it silently trains the generator
       backwards.
    2. Update the generator through a FRESH discriminator pass. The
       discriminator's parameters are not stepped here, but the gradient must
       flow through it to reach the generator -- the discriminator is the
       generator's only source of information about what "real" looks like.
    """

    def __init__(self, latent_dim, n_features, hidden=(64, 64), rng=None,
                 loss=None, output_activation=None):
        super().__init__()
        rng = check_random_state(rng)
        self.generator = Generator(latent_dim, n_features, hidden, rng=rng,
                                   output_activation=output_activation)
        self.discriminator = Discriminator(n_features, hidden, rng=rng)
        self.loss = loss if loss is not None else NonSaturatingGANLoss()
        self.latent_dim = latent_dim

    def step(self, real_batch, opt_g, opt_d, n_critic=1, grad_clip=None):
        """One alternating update. Returns (d_loss, g_loss)."""
        real = Tensor._wrap(real_batch)
        n = real.shape[0]

        for _ in range(n_critic):
            opt_d.zero_grad()
            fake = self.generator(self.generator.sample_latent(n))
            # detach: this loss must never move the generator
            d_loss = self.loss.discriminator_loss(
                self.discriminator(real), self.discriminator(fake.detach()))
            d_loss.backward()
            opt_d.step()
            if grad_clip is not None:
                for p in self.discriminator.parameters():
                    np.clip(p.data, -grad_clip, grad_clip, out=p.data)

        opt_g.zero_grad()
        fake = self.generator(self.generator.sample_latent(n))
        # no detach: the gradient must pass THROUGH the discriminator
        g_loss = self.loss.generator_loss(self.discriminator(fake))
        g_loss.backward()
        opt_g.step()
        return float(d_loss.item()), float(g_loss.item())

    def sample(self, n, rng=None):
        return self.generator.sample(n, rng=rng)

    def forward(self, z):
        return self.generator(z)


__all__ = ["Generator", "Discriminator", "GANLoss", "NonSaturatingGANLoss",
           "WGANLoss", "GAN"]
