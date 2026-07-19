"""IoU-family losses for bounding boxes: the metric IS the objective"""
import numpy as np
from ..autograd import Tensor
from .module import Module


def _box_ious(pred, target):
    """Return (iou, enclosing corners, centers, wh) for two batches of xyxy boxes."""
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
    inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
    inter = inter_w * inter_h
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_t = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
    union = area_p + area_t - inter + 1e-9
    iou = inter / union
    return iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2), union


class BoxRegressionLoss(Module):
    """IoU-family losses for bounding boxes: the metric IS the objective
    (Rezatofighi 2019; Zheng 2020).

    Regressing box corners with squared error optimises the wrong thing -- two
    boxes can have equal corner error yet wildly different overlap, and non-
    overlapping boxes give no gradient at all. The IoU family fixes this:

    * ``iou``  -- 1 - overlap. Zero gradient when boxes are disjoint.
    * ``giou`` -- adds the area of the smallest ENCLOSING box, so disjoint boxes
      still get pulled together.
    * ``diou`` -- penalises the distance between box CENTRES: faster, and it
      converges even when one box contains the other.
    * ``ciou`` -- diou plus an ASPECT-RATIO term, the most complete of the four.

    Boxes are ``(x1, y1, x2, y2)``. Returns the mean loss.
    """

    def __init__(self, mode="ciou"):
        super().__init__()
        assert mode in ("iou", "giou", "diou", "ciou")
        self.mode = mode

    def forward(self, pred, target):
        pred = np.asarray(Tensor._wrap(pred).data, dtype=float)
        target = np.asarray(Tensor._wrap(target).data, dtype=float)
        iou, (px1, py1, px2, py2, tx1, ty1, tx2, ty2), _ = _box_ious(pred, target)
        loss = 1.0 - iou
        if self.mode == "iou":
            return Tensor(float(loss.mean()))
        # smallest enclosing box
        cx1 = np.minimum(px1, tx1); cy1 = np.minimum(py1, ty1)
        cx2 = np.maximum(px2, tx2); cy2 = np.maximum(py2, ty2)
        if self.mode == "giou":
            enc = (cx2 - cx1) * (cy2 - cy1) + 1e-9
            area_p = (px2 - px1) * (py2 - py1)
            area_t = (tx2 - tx1) * (ty2 - ty1)
            inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
            inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
            union = area_p + area_t - inter_w * inter_h + 1e-9
            giou = iou - (enc - union) / enc          # penalise the empty enclosure
            return Tensor(float((1.0 - giou).mean()))
        # center distance (DIoU / CIoU)
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        center_d = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        diag = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-9
        diou = iou - center_d / diag
        if self.mode == "diou":
            return Tensor(float((1.0 - diou).mean()))
        pw, ph = np.clip(px2 - px1, 1e-9, None), np.clip(py2 - py1, 1e-9, None)
        tw, th = np.clip(tx2 - tx1, 1e-9, None), np.clip(ty2 - ty1, 1e-9, None)
        v = (4 / np.pi ** 2) * (np.arctan(tw / th) - np.arctan(pw / ph)) ** 2
        alpha = v / (1 - iou + v + 1e-9)
        ciou = diou - alpha * v
        return Tensor(float((1.0 - ciou).mean()))


__all__ = ["BoxRegressionLoss"]
