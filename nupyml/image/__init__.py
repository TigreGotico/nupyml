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
from .vision import (harris_corners, corner_peaks, canny, match_template,
                     lucas_kanade, chamfer_distance_transform,
                     non_max_suppression, slic)
from .gaussian_pyramid import gaussian_pyramid
from .laplacian_pyramid import laplacian_pyramid
from .pyramid_reconstruct import pyramid_reconstruct
from .watershed import watershed
from .active_contour import active_contour
from .orb_descriptors import orb_descriptors
from .match_descriptors import match_descriptors
from .bag_of_visual_words import BagOfVisualWords
from .seam_carving import seam_carving
from .sift import sift
from .horn_schunck import horn_schunck
from .felzenszwalb import felzenszwalb
from .estimate_homography import estimate_homography
from .ransac_homography import ransac_homography
from .viola_jones import ViolaJones
from .fast_corners import fast_corners
from .blob_detection import blob_detection
from .hough_circles import hough_circles
from .fundamental_matrix import fundamental_matrix
from .ransac_fundamental import ransac_fundamental
from .mean_shift_segmentation import MeanShiftSegmentation
from .bilateral import bilateral_filter
from .non_local_means import non_local_means
from .total_variation import total_variation_denoise
from .anisotropic_diffusion import anisotropic_diffusion
from .graph_cut import graph_cut_segmentation
from .hu_moments import hu_moments

__all__ = ["histogram_of_oriented_gradients", "local_binary_pattern",
           "graycomatrix", "haralick_features", "gabor_kernel", "gabor_features",
           "integral_image", "rectangle_sum", "label", "erosion", "dilation",
           "opening", "closing", "hough_line",
           "harris_corners", "corner_peaks", "canny", "match_template",
           "lucas_kanade", "chamfer_distance_transform", "non_max_suppression",
           "slic",
           "gaussian_pyramid", "laplacian_pyramid", "pyramid_reconstruct",
           "watershed", "active_contour", "orb_descriptors", "match_descriptors",
           "BagOfVisualWords", "seam_carving",
           "sift", "horn_schunck", "felzenszwalb", "estimate_homography",
           "ransac_homography", "ViolaJones",
           "fast_corners", "blob_detection", "hough_circles",
           "fundamental_matrix", "ransac_fundamental", "MeanShiftSegmentation",
           "bilateral_filter", "non_local_means", "total_variation_denoise",
           "anisotropic_diffusion", "graph_cut_segmentation", "hu_moments"]
