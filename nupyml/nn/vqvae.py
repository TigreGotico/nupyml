"""VQ-VAE: a latent space of symbols rather than numbers.

THE IDEA
--------
A VAE's latent is a point in continuous space. A VQ-VAE's is an INDEX into a
learned codebook of vectors. The encoder produces a vector, and the layer snaps
it to the nearest entry::

    z_e(x) -----> nearest codebook entry e_k -----> z_q(x) = e_k
                  (a lookup, not a projection)

So the model compresses an image not to 32 floats but to 32 SYMBOLS drawn from a
vocabulary of, say, 512. That is the whole point, and it buys two things:

* **No posterior collapse.** A VAE's KL term actively pushes q(z|x) toward the
  prior, and if the decoder is strong enough it wins: the latent carries nothing
  and the model ignores it. There is no such pressure here -- the code is a
  discrete choice, and a choice cannot be "collapsed to the prior" by degrees.
* **The latent becomes a sequence of tokens**, which a second model can then
  learn a prior over -- exactly what makes VQ-VAE the front half of modern
  discrete-latent generative pipelines.

THE PROBLEM: ARGMIN HAS NO GRADIENT
-----------------------------------
Picking the nearest codebook entry is a step function of the encoder's output.
Its derivative is zero almost everywhere and undefined at the boundaries, so the
tape simply stops. The encoder would receive nothing and never learn.

THE FIX: THE STRAIGHT-THROUGH ESTIMATOR
---------------------------------------
Pass the quantized value forward, but pretend the quantizer was the identity when
going backward -- copy the decoder's gradient straight onto the encoder's output::

    z_q = z_e + (quantize(z_e) - z_e).detach()

Read it carefully; the trick is entirely in the ``detach``. In the FORWARD pass
the ``z_e`` terms cancel and the value is exactly ``quantize(z_e)``. In the
BACKWARD pass the detached term contributes no gradient, so ``d z_q / d z_e = 1``
and the gradient flows through untouched.

This is a deliberate LIE -- the true Jacobian is not the identity, and that is
precisely what makes it usable. It is biased, but the bias is tolerable when
``z_e`` is already close to its code, which the commitment loss below enforces.

THE THREE LOSS TERMS
--------------------
The straight-through path trains the encoder and decoder, but it routes around
the codebook entirely -- so the codebook itself would never move. Hence::

    reconstruction  ||decode(z_q) - x||^2      encoder + decoder (via the STE)
    codebook        ||sg[z_e] - e||^2          moves each CODE toward its inputs
    commitment      beta * ||z_e - sg[e]||^2   moves the ENCODER toward its code

``sg`` is stop-gradient (``detach``). The two middle terms look symmetric but are
opposites: each freezes one side and pulls the other. Splitting them is what lets
``beta`` set the balance -- how much the encoder must commit to a code versus how
freely the codebook chases the encoder. Without the commitment term the encoder's
output can grow without bound, outrunning a codebook that never catches up.

van den Oord, Vinyals & Kavukcuoglu (2017).
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .autoencoder import _mlp
from .layers import Sigmoid


class VectorQuantizer(Module):
    """Snap each input vector to its nearest codebook entry.

    OPTIMIZATION: the nearest-code search is the classic squared-distance
    expansion, so the whole batch resolves in one matmul rather than a Python
    loop over codes::

        ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2

    The ``||z||^2`` column is constant across codes, so it cannot change which
    code is nearest and is dropped -- argmin only needs the terms that vary. That
    leaves one (batch x dim) @ (dim x codes) product, which BLAS does at full
    speed, instead of ``n_codes`` separate subtractions.
    """

    def __init__(self, n_codes, code_dim, beta=0.25, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        # uniform init over a small range: codes must start spread out, or the
        # nearest-neighbour search sends every input to the same entry and the
        # rest of the codebook never receives a gradient (a dead codebook)
        self.codebook = Parameter(rng.uniform(-1.0 / n_codes, 1.0 / n_codes,
                                              size=(n_codes, code_dim)))

    def lookup(self, indices):
        return Tensor(self.codebook.data[np.asarray(indices)])

    def encode_indices(self, z_e):
        """Which code is nearest, for each row. No gradient: this is an argmin."""
        z = np.asarray(z_e.data if isinstance(z_e, Tensor) else z_e)
        e = self.codebook.data
        # the dropped ||z||^2 term is constant per row -- see the class docstring
        dist = (e ** 2).sum(axis=1) - 2.0 * (z @ e.T)
        return np.argmin(dist, axis=1)

    def forward(self, z_e):
        z_e = Tensor._wrap(z_e)
        indices = self.encode_indices(z_e)
        codes = self._gathered(indices)

        # each term freezes one side and pulls the other -- see the module docstring
        codebook_loss = ((z_e.detach() - codes) ** 2).mean()
        commitment_loss = ((z_e - codes.detach()) ** 2).mean() * self.beta

        # the straight-through estimator: quantized forward, identity backward
        z_q = z_e + (codes - z_e).detach()
        return z_q, codebook_loss + commitment_loss, indices

    def _gathered(self, indices):
        """The chosen codes, still attached to the codebook Parameter.

        Indexing ``self.codebook.data`` would drop off the tape, and then the
        codebook loss could never move the codes it is supposed to train.
        """
        onehot = Tensor(np.eye(self.n_codes)[indices])
        return onehot @ self.codebook


class VQVAE(Module):
    """An autoencoder whose bottleneck is a codebook lookup.

    ``perplexity`` is the diagnostic to watch. It reports the effective number of
    codes in use: ``exp(entropy of the code histogram)``. A perplexity of 2 with
    a 512-entry codebook means 510 codes are dead -- the model has quietly become
    a 1-bit autoencoder, and reconstruction will be capped no matter how long it
    trains. Codebook collapse is this architecture's characteristic failure, the
    way posterior collapse is the VAE's.
    """

    def __init__(self, n_features, code_dim=8, n_codes=32, hidden=(64,),
                 beta=0.25, rng=None, output_activation=Sigmoid):
        super().__init__()
        rng = check_random_state(rng)
        self.encoder = _mlp([n_features, *hidden, code_dim], rng)
        self.quantizer = VectorQuantizer(n_codes, code_dim, beta=beta, rng=rng)
        self.decoder = _mlp([code_dim, *reversed(hidden), n_features], rng,
                            final_activation=output_activation)

    def forward(self, x):
        """Return (reconstruction, vq_loss, code indices)."""
        z_e = self.encoder(Tensor._wrap(x))
        z_q, vq_loss, indices = self.quantizer(z_e)
        return self.decoder(z_q), vq_loss, indices

    def loss(self, x, recon_loss=None):
        recon, vq_loss, _ = self(x)
        target = Tensor._wrap(x).detach()
        rec = ((recon - target) ** 2).mean() if recon_loss is None \
            else recon_loss(recon, target)
        return rec + vq_loss

    def perplexity(self, x):
        """Effective codebook usage. Far below n_codes means codes are dead."""
        _, _, indices = self(x)
        counts = np.bincount(indices, minlength=self.quantizer.n_codes)
        p = counts[counts > 0] / counts.sum()
        return float(np.exp(-(p * np.log(p)).sum()))

    def decode_indices(self, indices):
        """Reconstruct straight from symbols -- no encoder involved.

        This is what makes the discrete latent worth having: train a prior over
        these index sequences, sample from it, and decode the result.
        """
        return self.decoder(self.quantizer.lookup(indices)).data


__all__ = ["VectorQuantizer", "VQVAE"]
