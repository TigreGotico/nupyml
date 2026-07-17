"""nupyml: machine learning and neural networks, in readable numpy.

Every algorithm here is written to be READ. There is no compiled extension to
disappear into: if you can read numpy, you can read every line of every model.
Optimizations are not avoided, but they are explained rather than assumed --
where the fast way differs from the obvious way, the code says why.

WHERE TO START
--------------
Each module's docstring explains the FAMILY -- what it assumes, when it is the
wrong tool, and why the algorithms in it differ from one another. Read those
first; the class docstrings then cover the specific method, and comments cover
the implementation.

A reasonable path through the library:

1. ``linear_model``  -- the foundation. What a loss is, what a penalty does, and
   why L1 selects while L2 only shrinks.
2. ``metrics``       -- how anything is judged, and why accuracy usually is not
   the answer.
3. ``model_selection`` -- how not to fool yourself. Arguably the most important
   module here.
4. ``tree`` then ``ensemble`` -- greedy splitting, then the bagging/boosting
   split: cutting variance versus cutting bias.
5. ``cluster`` and ``mixture`` -- learning without labels, and EM.
6. ``svm``           -- margins, the kernel trick, and a real constrained
   optimizer worked through.
7. ``autograd`` then ``nn`` -- backpropagation from first principles, then the
   layers built on it.

WHERE THE INTERESTING IMPLEMENTATION IS
---------------------------------------
* ``autograd/tensor.py``     -- reverse-mode autodiff in ~400 lines. Why reverse
  mode, why a tape, why topological order.
* ``svm/__init__.py``        -- SMO with second-order working-set selection.
* ``ensemble/_hist_gb.py``   -- histogram trees and the subtraction trick.
* ``decomposition/__init__.py`` -- three routes to the same eigenvectors, and
  what each trades.
* ``autograd/functional.py`` -- convolution as a matmul; numerical stability of
  softmax and log-softmax.

CONVENTIONS
-----------
Every estimator follows the same contract -- ``fit`` / ``predict`` /
``transform``, parameters in ``__init__`` and never modified there, learned
attributes ending in an underscore -- so they compose in a ``Pipeline`` and can
be tuned by the ``*SearchCV`` estimators. ``utils.estimator_checks.check_estimator``
enforces it, and is worth reading as a statement of what the contract IS.

Only numpy and scipy are required at runtime.
"""
from .base import BaseEstimator, clone

__version__ = "0.1.0a1"
__all__ = ["BaseEstimator", "clone", "__version__"]
