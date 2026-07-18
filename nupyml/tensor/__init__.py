"""Tensor decompositions -- multi-way generalisations of PCA and NMF.

* **CP / PARAFAC** (``cp_decomposition``) -- a sum of rank-1 tensors; uniquely
  identifiable, so its factors are interpretable.
* **Tucker / HOSVD** (``tucker_decomposition``) -- a core tensor times a factor
  matrix per mode; PCA per mode, with full inter-mode interaction.
* **Tensor-train** (``tensor_train``) -- a chain of 3-way cores; storage linear in
  the order, taming the curse of dimensionality.
* **Non-negative CP** (``non_negative_cp``) -- CP with non-negative, parts-based
  factors.

Plus the tensor-algebra helpers (``unfold``/``fold``/``mode_dot``/
``khatri_rao``). numpy only.
"""
from .decomposition import (unfold, fold, mode_dot, khatri_rao,
                            cp_decomposition, cp_to_tensor,
                            tucker_decomposition, tucker_to_tensor,
                            tensor_train, tt_to_tensor, non_negative_cp)

__all__ = ["unfold", "fold", "mode_dot", "khatri_rao",
           "cp_decomposition", "cp_to_tensor",
           "tucker_decomposition", "tucker_to_tensor",
           "tensor_train", "tt_to_tensor", "non_negative_cp"]
