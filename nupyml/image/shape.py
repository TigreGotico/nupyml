"""Shape and region tools: integral image, connected components, morphology,
and the Hough transform.

Where ``features`` describes texture, these extract STRUCTURE: fast area sums,
which pixels form one object, cleaning up a binary mask, and finding the lines
hidden in an edge map.
"""
import numpy as np


def integral_image(image):
    """Summed-area table: pre-compute so ANY rectangle's sum is O(1).

    THE TRICK
    ---------
    ``ii[y, x]`` holds the sum of all pixels above-and-left of (y, x). Then the
    sum over any axis-aligned rectangle is four array lookups::

        S = ii[y1,x1] - ii[y0-1,x1] - ii[y1,x0-1] + ii[y0-1,x0-1]

    -- independent of the rectangle's size. This is what makes Viola-Jones face
    detection real-time: thousands of box features per window, each evaluated in
    constant time. Build it once, query forever.
    """
    image = np.asarray(image, dtype=float)
    return image.cumsum(axis=0).cumsum(axis=1)


def rectangle_sum(ii, y0, x0, y1, x1):
    """Sum over the inclusive rectangle [y0:y1, x0:x1] from an integral image."""
    total = ii[y1, x1]
    if y0 > 0:
        total -= ii[y0 - 1, x1]
    if x0 > 0:
        total -= ii[y1, x0 - 1]
    if y0 > 0 and x0 > 0:
        total += ii[y0 - 1, x0 - 1]
    return total


def label(binary, connectivity=1):
    """Connected-component labelling: give each blob of True pixels its own id.

    A binary mask says WHICH pixels are foreground; this says which of them
    belong to the SAME object. Two foreground pixels share a label iff a path of
    foreground pixels connects them (4-connectivity for ``connectivity=1``,
    8-connectivity for ``2``). Implemented as a flood fill from each unlabelled
    foreground pixel. Returns (labels, n_components).
    """
    binary = np.asarray(binary, dtype=bool)
    labels = np.zeros(binary.shape, dtype=int)
    if connectivity == 1:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)]
    rows, cols = binary.shape
    current = 0
    for i in range(rows):
        for j in range(cols):
            if binary[i, j] and labels[i, j] == 0:
                current += 1
                stack = [(i, j)]
                labels[i, j] = current
                while stack:                    # iterative flood fill
                    y, x = stack.pop()
                    for dy, dx in nbrs:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < rows and 0 <= nx < cols
                                and binary[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def _binary_op(binary, selem, op):
    from scipy.ndimage import binary_erosion, binary_dilation
    binary = np.asarray(binary, dtype=bool)
    if selem is None:
        selem = np.ones((3, 3), dtype=bool)
    return (binary_erosion if op == "erode" else binary_dilation)(
        binary, structure=selem)


def erosion(binary, selem=None):
    """Shrink foreground: a pixel survives only if ALL its neighbours are on.

    Erosion strips the boundary layer off every blob -- it removes small specks
    and thin protrusions, and is the "AND over the neighbourhood" half of
    morphology. The structuring element ``selem`` sets which neighbours count.
    """
    return _binary_op(binary, selem, "erode")


def dilation(binary, selem=None):
    """Grow foreground: a pixel turns on if ANY neighbour is on.

    Dilation is erosion's dual -- it fills small holes and joins nearby blobs, the
    "OR over the neighbourhood" operation.
    """
    return _binary_op(binary, selem, "dilate")


def opening(binary, selem=None):
    """Erosion then dilation: remove small objects, keep large ones' shape.

    Opening deletes anything too small to survive the erosion, then restores the
    size of what remained -- the standard way to clean salt noise from a mask
    without shrinking the real objects.
    """
    return dilation(erosion(binary, selem), selem)


def closing(binary, selem=None):
    """Dilation then erosion: fill small holes, keep the outer shape.

    Closing is opening's dual -- it seals pinholes and hairline gaps inside
    objects while leaving their outer boundary where it was.
    """
    return erosion(dilation(binary, selem), selem)


def hough_line(edges, n_angles=180):
    """Hough transform: find straight lines by VOTING in parameter space.

    THE IDEA
    --------
    A line is ``x*cos(theta) + y*sin(theta) = rho``. Each edge pixel could lie on
    many lines -- one for every angle -- so it VOTES for the whole (rho, theta)
    curve of lines through it. Where many edge pixels' curves intersect in the
    accumulator, they agree on a single (rho, theta): that peak IS a line in the
    image. Turning "which pixels are collinear?" (a hard search) into "where is
    the accumulator brightest?" (a simple max) is the whole trick.

    Returns (accumulator, thetas, rhos).
    """
    edges = np.asarray(edges, dtype=bool)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_angles)
    diag = int(np.ceil(np.hypot(*edges.shape)))
    rhos = np.arange(-diag, diag + 1)
    acc = np.zeros((len(rhos), n_angles), dtype=int)
    ys, xs = np.nonzero(edges)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for y, x in zip(ys, xs):
        rho = np.round(x * cos_t + y * sin_t).astype(int) + diag
        for k in range(n_angles):
            acc[rho[k], k] += 1               # this pixel votes for every angle
    return acc, thetas, rhos


__all__ = ["integral_image", "rectangle_sum", "label", "erosion", "dilation",
           "opening", "closing", "hough_line"]
