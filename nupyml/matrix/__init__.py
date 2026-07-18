"""Low-rank matrix recovery.

* ``RobustPCA`` -- decompose a matrix into low-rank + sparse (outlier-robust PCA).
* ``SoftImpute`` -- complete a matrix with missing entries, assuming low rank.
* ``singular_value_threshold`` -- the nuclear-norm proximal operator both use.

numpy only.
"""
from .recovery import singular_value_threshold, RobustPCA, SoftImpute

__all__ = ["singular_value_threshold", "RobustPCA", "SoftImpute"]
