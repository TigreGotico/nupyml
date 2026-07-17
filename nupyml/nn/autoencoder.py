"""Autoencoders: learning a representation by reconstructing the input.

THE IDEA
--------
Squeeze the input through a bottleneck and ask for it back::

    x  ->  encoder  ->  z  ->  decoder  ->  x_hat

The only supervision is the input itself, so this is self-supervised. If ``z`` is
smaller than ``x`` the network cannot simply copy, and the only way to reconstruct
is to discover the structure the data actually has.

WHY NOT JUST USE PCA
--------------------
A linear autoencoder with squared error learns exactly the PCA subspace -- the
same span, though not necessarily the same orthogonal axes. So a bare
autoencoder adds nothing over PCA unless the encoder is NON-LINEAR, in which case
it can follow a curved manifold that no linear projection can (see
``nupyml.manifold`` for the same argument from the other side).

THE FAILURE MODE, AND THE FAMILY IT SPAWNED
-------------------------------------------
The objective has a trivial cheat: learn the identity. With a wide enough
bottleneck, the network copies its input, scores perfectly, and has learned
nothing. Every variant here is a different way of forbidding that:

``AutoEncoder``
    Forbid it by making the bottleneck NARROW. Simple, and the capacity is a
    blunt instrument.
``DenoisingAutoEncoder``
    Corrupt the input, ask for the CLEAN version. Copying is now useless -- it
    would reproduce the noise -- so the network must model what the data has in
    common. Arguably the most robust of the family for the least work.
``SparseAutoEncoder``
    Allow a wide bottleneck but require most units to be OFF. Capacity is limited
    by activity rather than width, so the code can be overcomplete and still
    forced to specialise.
``ContractiveAutoEncoder``
    Penalise the encoder's sensitivity directly, so nearby inputs must map to
    nearby codes. Explicitly asks for the invariance the others get indirectly.

WHAT THIS FAMILY CANNOT DO
--------------------------
Nothing here is generative. The decoder maps codes to data, but there is no
distribution over ``z``, so there is no principled way to SAMPLE one: pick a
random z and the decoder will emit nonsense, because the region you picked from
was never trained on. That gap is exactly what ``VAE`` fills.
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Sequential
from .layers import Linear, ReLU, Sigmoid


def _mlp(sizes, rng, activation=ReLU, final_activation=None):
    """Build a plain MLP: Linear/activation pairs, optional final activation."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng=rng))
        if i < len(sizes) - 2:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation())
    return Sequential(*layers)


class AutoEncoder(Module):
    """Encoder and decoder trained to reconstruct the input through a bottleneck.

    ``hidden`` lists the encoder's layer widths; the decoder mirrors them. The
    final ``latent_dim`` is the bottleneck -- the only thing standing between the
    network and the identity function.

    Reconstruct with ``forward``; get the representation with ``encode``, which
    is usually the point: the decoder is scaffolding discarded after training.
    """

    def __init__(self, n_features, latent_dim, hidden=(64,), rng=None,
                 output_activation=Sigmoid):
        super().__init__()
        rng = check_random_state(rng)
        self.encoder = _mlp([n_features, *hidden, latent_dim], rng)
        self.decoder = _mlp([latent_dim, *reversed(hidden), n_features], rng,
                            final_activation=output_activation)
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(Tensor._wrap(x))

    def decode(self, z):
        return self.decoder(Tensor._wrap(z))

    def forward(self, x):
        return self.decode(self.encode(x))


class DenoisingAutoEncoder(AutoEncoder):
    """Corrupt the input, reconstruct the original.

    The training pair becomes ``(corrupt(x), x)`` rather than ``(x, x)``, which
    destroys the identity shortcut: copying the input would faithfully reproduce
    the noise, which is exactly what the loss punishes. To remove noise the
    network must know what the data looks like without it.

    A useful way to see it: the model learns to project a corrupted point back
    onto the data manifold, which means it has learned the manifold -- and it
    learns more of it with a wider bottleneck, not less, so the usual capacity
    tradeoff softens.

    ``noise="gaussian"`` adds noise; ``noise="masking"`` zeroes a random fraction
    of inputs, which is the same idea that later became masked language
    modelling.
    """

    def __init__(self, n_features, latent_dim, hidden=(64,), noise="gaussian",
                 noise_level=0.3, rng=None, output_activation=Sigmoid):
        super().__init__(n_features, latent_dim, hidden, rng, output_activation)
        self.noise = noise
        self.noise_level = noise_level
        self._rng = check_random_state(rng)

    def corrupt(self, x):
        """Only during training: at eval time a clean input is what we want."""
        if not self.training:
            return x
        data = x.data if isinstance(x, Tensor) else np.asarray(x)
        if self.noise == "gaussian":
            return Tensor(data + self._rng.normal(scale=self.noise_level,
                                                  size=data.shape))
        if self.noise == "masking":
            keep = self._rng.uniform(size=data.shape) >= self.noise_level
            return Tensor(data * keep)
        raise ValueError(f"Unknown noise: {self.noise!r}")

    def forward(self, x):
        return self.decode(self.encode(self.corrupt(x)))


class SparseAutoEncoder(AutoEncoder):
    """Wide bottleneck, but most units must stay off.

    Capacity is limited by ACTIVITY rather than width. The penalty is a KL
    divergence between each unit's average activation and a small target ``rho``
    (typically 0.05), summed over units::

        KL(rho || rho_hat) = rho*log(rho/rho_hat) + (1-rho)*log((1-rho)/(1-rho_hat))

    Read it as: treat each unit's mean activation as a coin, and require that
    coin to land heads 5% of the time. KL rather than an L1 penalty because it
    grows steeply as a unit approaches always-on OR always-off -- it wants the
    right rate, not merely a small one.

    The payoff is that the code may be OVERCOMPLETE (more units than inputs) and
    still learn something: each unit specialises to a feature that is rarely
    present, which is what makes sparse codes interpretable.
    """

    def __init__(self, n_features, latent_dim, hidden=(64,), rho=0.05,
                 beta=1.0, rng=None, output_activation=Sigmoid):
        super().__init__(n_features, latent_dim, hidden, rng, output_activation)
        # a sigmoid code keeps activations in [0, 1] so they read as rates
        self.encoder = Sequential(*self.encoder.layers, Sigmoid())
        self.rho = rho
        self.beta = beta

    def sparsity_penalty(self, code):
        rho_hat = code.mean(axis=0).clip(1e-6, 1 - 1e-6)
        rho = self.rho
        kl = (rho * (np.log(rho) - rho_hat.log())
              + (1 - rho) * (np.log(1 - rho) - (1.0 - rho_hat).log()))
        return kl.sum() * self.beta

    def loss(self, x, loss_fn):
        """Reconstruction plus the sparsity penalty on the code."""
        x = Tensor._wrap(x)
        code = self.encode(x)
        recon = self.decode(code)
        return loss_fn(recon, x.detach()) + self.sparsity_penalty(code)


class VAE(Module):
    """Variational autoencoder: an autoencoder that is actually a generative model.

    THE PROBLEM WITH A PLAIN AUTOENCODER
    ------------------------------------
    Its latent space has holes. Training only ever visits the codes of real
    inputs, so the regions between them are undefined, and sampling a random z
    gives garbage. There is no distribution over z at all -- so nothing to sample
    FROM.

    THE FIX: ENCODE A DISTRIBUTION, NOT A POINT
    -------------------------------------------
    The encoder outputs a mean and a variance, and z is DRAWN from that gaussian.
    Two consequences follow:

    * Each input now maps to a REGION, and nearby codes must decode sensibly,
      because they are all sampled during training. The holes close.
    * A KL term pulls every such region toward a standard normal, so the codes
      collectively fill N(0, I) rather than drifting apart. Now sampling z ~
      N(0, I) and decoding produces something plausible -- the model generates.

    THE OBJECTIVE (ELBO)
    --------------------
    The thing we want, ``log p(x)``, is intractable: it needs an integral over
    every z. So maximise a lower bound on it instead::

        log p(x)  >=  E_q[log p(x|z)]  -  KL(q(z|x) || p(z))
                      \\_____________/     \\__________________/
                       reconstruction         regularisation

    The gap between the bound and the truth is exactly ``KL(q || true posterior)``
    -- so tightening the bound also makes the encoder a better posterior
    approximation. Maximising the ELBO does both jobs at once, which is the
    elegance of the thing.

    Read the two terms as a tug of war. Reconstruction wants each input's code
    to be distinct and precise (tiny variance, spread-out means). KL wants every
    code to be N(0, I) and thus indistinguishable. The balance is a latent space
    that is both informative and complete.

    THE REPARAMETERIZATION TRICK
    ----------------------------
    ``z ~ N(mu, sigma^2)`` is a sampling step, and you cannot backpropagate
    through sampling -- the derivative of "draw a random number" is not defined,
    and mu and sigma are behind it.

    The trick is to move the randomness OUT of the path::

        z = mu + sigma * eps,   eps ~ N(0, 1)

    Identical distribution. But now ``eps`` is just an input -- a constant, as far
    as the tape is concerned -- and z is a deterministic, differentiable function
    of mu and sigma. The gradient flows straight through. This one substitution
    is what made VAEs trainable, and it is worth staring at until it is obvious.

    WHY log-variance
    ----------------
    The encoder emits ``log(sigma^2)``, not ``sigma``. Variance must be positive,
    and a network's output is not; exponentiating guarantees it without a clamp,
    keeps the gradient well-scaled across orders of magnitude, and makes the KL
    term's closed form cheap.

    Kingma & Welling (2013).
    """

    def __init__(self, n_features, latent_dim, hidden=(64,), rng=None,
                 output_activation=Sigmoid):
        super().__init__()
        rng = check_random_state(rng)
        self._rng = rng
        self.latent_dim = latent_dim
        enc_sizes = [n_features, *hidden]
        self.encoder_body = _mlp(enc_sizes, rng) if len(enc_sizes) > 1 \
            else Sequential()
        last = hidden[-1] if hidden else n_features
        # two heads: the encoder describes a distribution, not a point
        self.mu_head = Linear(last, latent_dim, rng=rng)
        self.logvar_head = Linear(last, latent_dim, rng=rng)
        self.decoder = _mlp([latent_dim, *reversed(hidden), n_features], rng,
                            final_activation=output_activation)

    def encode(self, x):
        """Return (mu, logvar): the parameters of q(z|x)."""
        h = self.encoder_body(Tensor._wrap(x))
        h = h.relu()
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        """``z = mu + sigma * eps``, with eps drawn outside the graph.

        At eval time the mean is returned directly: inference wants the code,
        not a sample from around it.
        """
        if not self.training:
            return mu
        std = (logvar * 0.5).exp()
        eps = self._rng.normal(size=mu.shape)
        return mu + std * Tensor(eps)          # eps is a constant to the tape

    def decode(self, z):
        return self.decoder(Tensor._wrap(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n_samples, rng=None):
        """Generate by decoding draws from the prior -- the whole point of a VAE."""
        rng = check_random_state(rng if rng is not None else self._rng)
        z = rng.normal(size=(n_samples, self.latent_dim))
        was_training = self.training
        self.eval()
        out = self.decode(Tensor(z)).data
        self.train(was_training)
        return out


def kl_divergence_normal(mu, logvar):
    """``KL(N(mu, sigma^2) || N(0, 1))``, in closed form.

    No sampling needed: for two gaussians the KL integral has an exact solution::

        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Read each term. It is zero when ``mu = 0`` and ``sigma = 1`` -- the prior --
    and grows if the mean drifts away (``mu^2``) or the variance is wrong in
    either direction (``sigma^2`` punishes too wide, ``-log(sigma^2)`` punishes
    too narrow). That two-sided pressure on the variance is what stops the
    encoder from shrinking sigma to zero and degenerating into a plain
    autoencoder.

    Summed over latent dimensions, averaged over the batch.
    """
    return (-(logvar + 1.0 - mu * mu - logvar.exp()) * 0.5).sum(axis=-1).mean()


class VAELoss(Module):
    """Negative ELBO: reconstruction + beta * KL.

    ``beta`` is the dial between the two forces, and gives the family its
    members:

    * ``beta = 1`` -- the true ELBO, a proper bound on ``log p(x)``.
    * ``beta > 1`` -- beta-VAE. Weight the KL harder and the code is squeezed
      toward the prior, forcing it to discard everything but the most
      independently-useful factors. That pressure is what encourages
      DISENTANGLEMENT, at the cost of blurrier reconstructions.
    * ``beta < 1`` -- sharper reconstructions, a less well-behaved latent space,
      and sampling degrades toward the plain autoencoder's failure.

    POSTERIOR COLLAPSE
    ------------------
    The KL term has a degenerate optimum: set ``q(z|x) = p(z)`` exactly, ignore
    z entirely, and pay zero KL. If the decoder is powerful enough to do well
    without z, that is what training finds -- the latent space goes unused and
    KL sits at 0. Seeing KL fall to zero early is the symptom, and annealing
    beta from 0 upward is the usual remedy.

    ``recon_loss="bce"`` treats outputs as Bernoulli probabilities (right for
    binary or [0,1] pixels); ``"mse"`` treats them as gaussian.
    """

    def __init__(self, beta=1.0, recon_loss="bce"):
        super().__init__()
        self.beta = beta
        self.recon_loss = recon_loss

    def forward(self, recon, target, mu, logvar):
        t = Tensor._wrap(target).detach()
        if self.recon_loss == "bce":
            p = recon.clip(1e-7, 1 - 1e-7)
            # summed over features, averaged over the batch: the ELBO is a
            # per-sample quantity, and averaging over features instead would
            # silently rescale the KL term against it
            rec = -(t * p.log() + (1.0 - t) * (1.0 - p).log()).sum(axis=-1).mean()
        elif self.recon_loss == "mse":
            rec = ((recon - t) ** 2).sum(axis=-1).mean()
        else:
            raise ValueError(f"Unknown recon_loss: {self.recon_loss!r}")
        return rec + kl_divergence_normal(mu, logvar) * self.beta


class ConditionalVAE(VAE):
    """A VAE that is told the class, so generation can be steered.

    The label is concatenated to both the encoder's input and the decoder's, so
    ``z`` no longer needs to carry class identity -- the decoder is told it. That
    frees the latent space to encode STYLE (the within-class variation), and it
    makes generation controllable: pick a class, draw a z, get that class.
    """

    def __init__(self, n_features, latent_dim, n_classes, hidden=(64,),
                 rng=None, output_activation=Sigmoid):
        super().__init__(n_features + n_classes, latent_dim, hidden, rng,
                         output_activation)
        rng = check_random_state(rng)
        self.n_classes = n_classes
        self.n_features = n_features
        # the decoder also sees the label, so rebuild it wider
        self.decoder = _mlp([latent_dim + n_classes, *reversed(hidden),
                             n_features], rng, final_activation=output_activation)

    def _onehot(self, y):
        y = np.asarray(y, dtype=np.int64)
        return Tensor(np.eye(self.n_classes)[y])

    def forward(self, x, y):
        x = Tensor._wrap(x)
        cond = self._onehot(y)
        mu, logvar = self.encode(Tensor.concatenate([x, cond], axis=1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(Tensor.concatenate([z, cond], axis=1))
        return recon, mu, logvar

    def sample(self, y, rng=None):
        """Generate examples of the classes named in ``y``."""
        rng = check_random_state(rng if rng is not None else self._rng)
        y = np.asarray(y, dtype=np.int64)
        z = rng.normal(size=(len(y), self.latent_dim))
        was_training = self.training
        self.eval()
        out = self.decode(Tensor.concatenate([Tensor(z), self._onehot(y)],
                                             axis=1)).data
        self.train(was_training)
        return out


__all__ = ["AutoEncoder", "DenoisingAutoEncoder", "SparseAutoEncoder", "VAE",
           "ConditionalVAE", "VAELoss", "kl_divergence_normal"]
