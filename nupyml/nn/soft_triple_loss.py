"""SoftTriple loss: multiple learnable centres per class for metric learning."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module, Parameter


class SoftTripleLoss(Module):
    """Give each class SEVERAL centres, not one (Qian et al., 2019).

    Softmax and centre-based metric losses assume each class is a single blob, but real
    classes have intra-class STRUCTURE -- "dog" spans many breeds. SoftTriple gives every
    class K learnable centres and scores a sample against a class by a softmax-weighted
    blend of its similarities to that class's centres, so a sample only has to match ONE
    facet of its class. A large-margin softmax over those blended scores then pulls
    samples toward their nearest centre while pushing other classes away. It captures
    multi-mode classes without the sampling headaches of triplet mining. Embeddings are
    assumed L2-normalised; ``n_centers`` centres per class.
    """

    def __init__(self, n_classes, dim, n_centers=2, gamma=10.0, margin=0.01,
                 scale=20.0, random_state=None):
        super().__init__()
        rng = np.random.RandomState(random_state)
        w = rng.randn(n_classes * n_centers, dim)
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-9
        self.centers = Parameter(w)
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.gamma = gamma
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        x = Tensor._wrap(embeddings)
        c = self.centers / ((self.centers * self.centers).sum(axis=1, keepdims=True)
                            + 1e-9).sqrt()
        sim = x @ c.transpose()                             # (N, C*K)
        n = sim.shape[0]
        # per-class score = softmax-weighted blend over that class's K centres
        blocks = []
        for cls in range(self.n_classes):
            s = sim[:, cls * self.n_centers:(cls + 1) * self.n_centers]   # (N, K)
            w = F.softmax(s * self.gamma, axis=1)
            blocks.append((w * s).sum(axis=1, keepdims=True))
        scores = Tensor.concatenate(blocks, axis=1)         # (N, C)
        # large-margin softmax: subtract the margin on the true class
        onehot = np.zeros((n, self.n_classes)); onehot[np.arange(n), target] = 1.0
        logits = (scores - Tensor(onehot) * self.margin) * self.scale
        logp = F.log_softmax(logits, axis=-1)
        return -logp[np.arange(n), target].mean()


__all__ = ["SoftTripleLoss"]
