"""Flood the intensity landscape from seed MARKERS until basins meet"""
import numpy as np
from .vision import harris_corners, corner_peaks, _gray


def watershed(image, markers, mask=None):
    """Flood the intensity landscape from seed MARKERS until basins meet
    (Meyer, 1994).

    Read the image as a topography -- bright = high. Start water rising from each
    labelled marker; as the level rises, each pixel joins the basin whose water
    reaches it first, and basins are kept from merging at the ridges between them.
    Those ridges become the segment boundaries. It is the classic way to split
    touching objects (cells, coins) once you can seed each one. Priority-flood
    implementation over the marker labels.
    """
    import heapq
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    markers = np.asarray(markers, int)
    labels = markers.copy()
    H, W = img.shape
    heap = []
    for y, x in zip(*np.where(markers > 0)):
        heapq.heappush(heap, (img[y, x], int(markers[y, x]), y, x))
    while heap:
        _, lab, y, x = heapq.heappop(heap)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0:
                if mask is not None and not mask[ny, nx]:
                    continue
                labels[ny, nx] = lab                     # join this basin
                heapq.heappush(heap, (img[ny, nx], lab, ny, nx))
    return labels


__all__ = ["watershed"]
