"""Self-distillation: match a SHARPENED teacher (Caron et al., 2021)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class DINOLoss(Module):
    """Self-distillation: match a SHARPENED teacher (Caron et al., 2021).

    DINO trains a network with no labels by distillation from itself: a "student" is
    pushed to match a "teacher" (an exponential moving average of the student) on
    different augmented views. Two tricks stop it collapsing to a constant -- the
    teacher output is SHARPENED (a low softmax temperature makes it confident) and
    CENTERED (a running mean is subtracted so no dimension dominates) -- and the
    student, at a higher temperature, is trained by cross-entropy to that target. The
    result is a powerful self-supervised representation. ``center`` is the running
    teacher mean.
    """

    def __init__(self, n_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.center = np.zeros((1, n_dim))

    def forward(self, student_out, teacher_out):
        s = Tensor._wrap(student_out)
        t = np.asarray(Tensor._wrap(teacher_out).data, float)
        # teacher: centre then sharpen (low temperature), as a fixed target
        teacher_soft = _softmax((t - self.center) / self.teacher_temp)
        logp = F.log_softmax(s * (1.0 / self.student_temp), axis=1)
        loss = -(Tensor(teacher_soft) * logp).sum(axis=1).mean()
        # update the centre with the batch mean (prevents collapse)
        self.center = (self.center_momentum * self.center
                       + (1 - self.center_momentum) * t.mean(axis=0, keepdims=True))
        return loss


__all__ = ["DINOLoss"]
