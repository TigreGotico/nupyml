"""Nonparametric regression: flexible models you can still read.

THE GAP THIS FILLS
------------------
Between a linear model and a tree there is a chasm:

* **Linear regression** is perfectly interpretable -- one coefficient per
  feature, and you can state what the model believes in a sentence. It is also
  wrong whenever the truth is not a straight line.
* **A random forest** fits nearly anything and explains nothing. Ask why it
  predicted 7 and the honest answer is "four hundred trees voted".

The methods here are flexible AND readable. That combination is not a
compromise, it is a design goal, and each one buys it differently:

* ``LOESS`` -- fit a little line near each query point. No global form at all.
* ``GAM`` -- sum of one smooth function per feature. You can PLOT what the model
  thinks each feature does; that plot IS the model.
* ``MARS`` -- automatically discover where the relationship bends, and represent
  it as hinges. Gives you an equation with breakpoints.
* ``RuleFit`` -- extract rules from a forest, then let the lasso pick a handful.
  A forest's accuracy, a short list of rules as the explanation.
* ``KernelRidge`` -- ridge regression in a kernel's feature space.

THE ONE ASSUMPTION THEY SHARE
-----------------------------
SMOOTHNESS: nearby inputs have similar outputs. That is a much weaker assumption
than linearity, but it is not nothing -- it is exactly what fails in high
dimensions.

THE CURSE, STATED PLAINLY
-------------------------
Every method here estimates the function locally, from the points nearby. In
high dimensions there are no points nearby. To capture 10% of the data in a
neighbourhood you need to span 10% of the range in 1D, 32% per side in 5D, and
63% per side in 10D -- at which point the "neighbourhood" is most of the space
and "local" means nothing at all.

So these methods are excellent up to a handful of dimensions and hopeless beyond
it. That is not a quality-of-implementation issue; it is geometry. Trees and
regularised linear models survive high dimensions precisely because they DO make
strong global assumptions. The additive structure in ``GAM`` and ``MARS`` is the
usual escape: assuming the function is a SUM of low-dimensional pieces sidesteps
the curse, at the price of that assumption being true.
"""
from ._loess import LOESS, NadarayaWatson, LocalLinearRegression
from ._gam import GAM, GAMClassifier, SplineTransformer, natural_cubic_basis
from ._mars import MARS
from ._rulefit import RuleFit, Rule
from ._kernel_ridge import KernelRidge
from ._robust import TotalLeastSquares, LTSRegressor, NonlinearLeastSquares
from ._bayes import RVMRegressor, BARTRegressor

__all__ = [
    "LOESS", "NadarayaWatson", "LocalLinearRegression",
    "GAM", "GAMClassifier", "SplineTransformer", "natural_cubic_basis",
    "MARS", "RuleFit", "Rule", "KernelRidge",
    "TotalLeastSquares", "LTSRegressor", "NonlinearLeastSquares",
    "RVMRegressor", "BARTRegressor",
]
