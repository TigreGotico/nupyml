"""Classic image / signal feature extraction -- the pre-deep-learning toolkit.

Two families:

* ``features`` -- descriptors that turn a pixel grid into a vector invariant to
  some nuisance: HOG (shape, invariant to illumination/small shifts), LBP
  (texture, invariant to monotonic lighting), GLCM + Haralick (texture
  statistics), and Gabor filter banks (oriented, multi-scale texture).
* ``shape`` -- structure tools: the integral image (constant-time box sums),
  connected-component labelling, binary morphology (erosion/dilation/opening/
  closing), and the Hough line transform (finding lines by voting).

These need no training, are interpretable, and work on tiny datasets -- exactly
where a CNN would overfit. Only numpy and scipy are used.
"""
from .features import (histogram_of_oriented_gradients, local_binary_pattern,
                       graycomatrix, haralick_features, gabor_kernel,
                       gabor_features)
from .shape import (integral_image, rectangle_sum, label, erosion, dilation,
                    opening, closing, hough_line)

__all__ = ["histogram_of_oriented_gradients", "local_binary_pattern",
           "graycomatrix", "haralick_features", "gabor_kernel", "gabor_features",
           "integral_image", "rectangle_sum", "label", "erosion", "dilation",
           "opening", "closing", "hough_line"]
