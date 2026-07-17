"""Self-supervised learning: labels invented from the data itself.

THE PROBLEM
-----------
Labels are expensive and data is free. Supervised learning needs the former;
everything here needs only the latter. The trick is to define a task whose answer
is already contained in the input, so the "label" costs nothing:

* **SimCLR** -- distort an example twice; the task is to recognise that the two
  distortions came from the same source.
* **Masked autoencoding** -- hide part of the input; the task is to fill it back
  in.

Neither task is useful in itself. Nobody wants a model that matches crops, or
that inpaints holes it created. The REPRESENTATION learned on the way is the
product, and the task is only a pretext -- which is the term of art for exactly
this reason.

WHY A PRETEXT TASK TEACHES ANYTHING
-----------------------------------
The pretext must be solvable only by understanding the content. To know that two
crops of a dog are the same photo, the network must represent "dog" rather than
pixels, because the pixels differ. To fill in a masked patch, it must know what
tends to go there.

Get the pretext wrong and the network finds a shortcut instead. If the two SimCLR
views differ only in brightness, matching them by mean colour works perfectly and
teaches nothing at all. The augmentations are not a detail of the method -- they
ARE the method, because they define what the model must become invariant to and
therefore what it is forced to encode.

THE CONTRAST WITH AN AUTOENCODER
--------------------------------
A plain autoencoder also needs no labels, so why is this different? Because it is
rewarded for reconstructing EVERYTHING, including noise, and its shortcut is to
copy. Both methods here forbid copying by construction: SimCLR compares two
different views (there is nothing to copy from), and a masked autoencoder hides
what it must predict. What cannot be copied must be understood.
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module
from .autoencoder import _mlp
from .metric_losses import InfoNCELoss


class SimCLR(Module):
    """Contrastive self-supervised learning.

    THE ARCHITECTURE
    ----------------
    Two pieces, and the split between them matters more than either::

        x --> encoder --> h --> projection head --> z
                          ^                        ^
                          keep this                throw this away

    The loss is applied to ``z``, but ``h`` is what you use downstream. This looks
    perverse and is one of the paper's real findings: the loss demands invariance
    to the augmentations, which means DISCARDING colour, orientation, scale --
    exactly the information a downstream task might want. The projection head
    absorbs that damage. ``h``, one layer upstream, keeps what ``z`` was forced to
    destroy, and is measurably the better representation.

    ``encode`` returns ``h``. That the class trains on something it then discards
    is the point, not an oversight.
    """

    def __init__(self, n_features, hidden=(64,), embed_dim=32, proj_dim=16,
                 temperature=0.5, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.encoder = _mlp([n_features, *hidden, embed_dim], rng)
        # deliberately small and non-linear: it exists to be discarded
        self.projection = _mlp([embed_dim, embed_dim, proj_dim], rng)
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def encode(self, x):
        """The representation you actually want -- before the projection head."""
        return self.encoder(Tensor._wrap(x))

    def project(self, x):
        return self.projection(self.encode(x))

    def forward(self, x):
        return self.encode(x)

    def loss(self, view_a, view_b):
        """NT-Xent over a batch of paired views.

        Row ``i`` of ``view_a`` and row ``i`` of ``view_b`` are the two views of
        the same example -- the positive pair. Every other row in the batch is a
        negative, which is why SimCLR is famously hungry for large batches: the
        number of negatives per step IS the batch size, and the task gets harder
        (and the representation better) the more of them there are.
        """
        return self.loss_fn(self.project(view_a), self.project(view_b))


class GaussianNoiseAugment:
    """The simplest augmentation that defines a non-trivial invariance.

    Real SimCLR uses crops, colour jitter and blur, whose composition is what
    makes the pretext hard. Additive noise is the tabular-data analogue: it asks
    the encoder to be invariant to small perturbations, which is a weaker demand
    but the same shape of demand.
    """

    def __init__(self, sigma=0.1, rng=None):
        self.sigma = sigma
        self.rng = check_random_state(rng)

    def __call__(self, x):
        return np.asarray(x) + self.rng.normal(0, self.sigma, np.shape(x))


class MaskingAugment:
    """Randomly zero a fraction of features.

    Note this is the SAME operation a denoising autoencoder uses as noise, and
    that a masked autoencoder uses as its task. The difference between the three
    methods is not the corruption -- it is what they ask of the result.
    """

    def __init__(self, p=0.3, rng=None):
        self.p = p
        self.rng = check_random_state(rng)

    def __call__(self, x):
        x = np.asarray(x)
        return x * (self.rng.uniform(size=x.shape) > self.p)


class MaskedAutoEncoder(Module):
    """Hide part of the input; predict what was hidden.

    THE ONE DETAIL THAT MATTERS
    ---------------------------
    The loss is computed ONLY on the masked positions. Scoring the visible ones
    too would let the model spend its capacity copying inputs it can already see
    -- work that is free and teaches nothing, and which would dominate the loss
    whenever the mask ratio is low.

    Restricting the loss to what was hidden means every unit of loss corresponds
    to an actual prediction. This is why MAE works with mask ratios as high as
    75%: the task must be hard enough that memorising local texture cannot solve
    it, and forcing genuinely global inference is the whole objective.
    """

    def __init__(self, n_features, hidden=(64,), embed_dim=32, mask_ratio=0.5,
                 rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self._rng = rng
        self.mask_ratio = mask_ratio
        self.encoder = _mlp([n_features, *hidden, embed_dim], rng)
        self.decoder = _mlp([embed_dim, *reversed(hidden), n_features], rng)

    def encode(self, x):
        return self.encoder(Tensor._wrap(x))

    def forward(self, x):
        return self.decoder(self.encode(x))

    def loss(self, x):
        x = np.asarray(x, dtype=np.float64)
        # True where a feature is hidden and must therefore be predicted
        mask = self._rng.uniform(size=x.shape) < self.mask_ratio
        recon = self(x * ~mask)

        n_masked = mask.sum()
        if n_masked == 0:
            return ((recon - Tensor(x)) ** 2).mean()
        # scored on the hidden positions only -- see the class docstring
        return (((recon - Tensor(x)) ** 2) * Tensor(mask.astype(np.float64))
                ).sum() / n_masked

    def reconstruct(self, x, mask):
        """Fill in ``mask``ed positions, keeping the visible ones as given."""
        x, mask = np.asarray(x, dtype=np.float64), np.asarray(mask, dtype=bool)
        pred = self(x * ~mask).data
        return np.where(mask, pred, x)


__all__ = ["SimCLR", "MaskedAutoEncoder", "GaussianNoiseAugment",
           "MaskingAugment"]
