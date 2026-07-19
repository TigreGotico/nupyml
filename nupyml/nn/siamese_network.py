"""Twin encoders with SHARED weights, for learning a similarity metric."""
from ..autograd import Tensor
from .module import Module


class SiameseNetwork(Module):
    """Twin encoders with SHARED weights, for learning a similarity metric.

    A Siamese network runs two inputs through the SAME encoder and compares their
    embeddings; trained with a contrastive or triplet loss (see ``metric_losses``)
    it learns an embedding where "same" pairs are close and "different" pairs far.
    Weight sharing is the essence -- both inputs are mapped by one function, so the
    geometry is consistent -- and it is the standard approach for verification and
    few-shot tasks where the classes at test time were never seen in training.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def parameters(self):
        return list(self.encoder.parameters())

    def encode(self, x):
        return self.encoder(Tensor._wrap(x) if not isinstance(x, Tensor) else x)

    def forward(self, x1, x2):
        return self.encode(x1), self.encode(x2)       # shared-weight twin pass


__all__ = ["SiameseNetwork"]
