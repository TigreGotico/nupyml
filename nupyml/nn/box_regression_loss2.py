"""Refined IoU box-regression losses: EIoU and SIoU (Zhang 2021; Gevorgyan 2022)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class BoxRegressionLoss2(Module):
    """Refined IoU box-regression losses: EIoU and SIoU (Zhang 2021; Gevorgyan 2022).

    CIoU couples the width/height into a single aspect-ratio term, whose gradient can
    push both the wrong way. ``eiou`` (Efficient IoU) instead penalises the width and
    height errors SEPARATELY, normalised by the enclosing box -- cleaner gradients and
    faster convergence. ``siou`` (SCYLLA IoU) adds an ANGLE cost that first aligns the
    boxes along the axis of their centre offset before closing the distance, which
    empirically trains detectors faster still. Boxes are ``(x1, y1, x2, y2)``; returns
    the mean loss.
    """

    def __init__(self, mode="eiou"):
        super().__init__()
        assert mode in ("eiou", "siou")
        self.mode = mode

    def forward(self, pred, target):
        pred = np.asarray(Tensor._wrap(pred).data, float)
        target = np.asarray(Tensor._wrap(target).data, float)
        px1, py1, px2, py2 = pred.T
        tx1, ty1, tx2, ty2 = target.T
        iw = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
        ih = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
        inter = iw * ih
        ap = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
        at = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
        iou = inter / (ap + at - inter + 1e-9)
        cx1 = np.minimum(px1, tx1); cy1 = np.minimum(py1, ty1)
        cx2 = np.maximum(px2, tx2); cy2 = np.maximum(py2, ty2)
        cw, ch = cx2 - cx1, cy2 - cy1
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        cdx, cdy = pcx - tcx, pcy - tcy
        diag = cw ** 2 + ch ** 2 + 1e-9
        if self.mode == "eiou":
            # centre distance + separate width and height distances (all normalised)
            dist = (cdx ** 2 + cdy ** 2) / diag
            wdist = ((px2 - px1) - (tx2 - tx1)) ** 2 / (cw ** 2 + 1e-9)
            hdist = ((py2 - py1) - (ty2 - ty1)) ** 2 / (ch ** 2 + 1e-9)
            loss = 1 - iou + dist + wdist + hdist
        else:                                             # siou
            sigma = np.sqrt(cdx ** 2 + cdy ** 2) + 1e-9
            sin_alpha = np.abs(cdy) / sigma
            angle_cost = 1 - 2 * np.sin(np.arcsin(np.clip(sin_alpha, 0, 1))
                                        - np.pi / 4) ** 2
            rho_x = (cdx / (cw + 1e-9)) ** 2; rho_y = (cdy / (ch + 1e-9)) ** 2
            gamma = 2 - angle_cost
            dist_cost = 2 - np.exp(-gamma * rho_x) - np.exp(-gamma * rho_y)
            loss = 1 - iou + 0.5 * (dist_cost + angle_cost * 0.1)
        return Tensor(float(loss.mean()))


__all__ = ["BoxRegressionLoss2"]
