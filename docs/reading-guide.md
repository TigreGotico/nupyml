# Reading guide

The whole library is meant to be read, but these are the files where the
*interesting* engineering lives — the places where the fast way differs from the
obvious way, and the code explains why.

| File | What it teaches |
|---|---|
| `autograd/tensor.py` | Reverse-mode autodiff in ~400 lines: why reverse mode, why a tape, why topological order. |
| `autograd/functional.py` | Convolution as a matmul (im2col); the numerically stable softmax and log-softmax. |
| `svm/__init__.py` | SMO with second-order working-set selection: what a "violating pair" is, why the working set is two variables, and how the incremental gradient keeps each step O(n). |
| `ensemble/_hist_gb.py` | Histogram gradient boosting: why `np.bincount` beats `np.add.at` by ~7×, and the subtraction trick (`parent - child == sibling`) that halves the work at every tree level. |
| `decomposition/__init__.py` | PCA by three routes to the same eigenvectors (`full`, `covariance_eigh`, `randomized`) — what each costs and the precision `covariance_eigh` trades away by squaring the condition number. |
| `cluster/__init__.py` | k-means: why the M-step is a scatter-add rather than a per-cluster mask, and what D² seeding buys. |
| `nn/vqvae.py` | The straight-through estimator: a deliberate lie about a derivative, and why it is the honest choice. |
| `evolutionary/strategies.py` | CMA-ES: a derivative-free quasi-Newton method that learns the landscape's shape into a covariance matrix. |
| `pgm/inference.py` | Variable elimination: pushing sums inside products so exact inference stays tractable. |
| `inference/conformal.py` | Why an exchangeability argument alone yields a finite-sample coverage guarantee — and where split conformal's fixed width falls short. |
| `inference/_advanced.py` | The No-U-Turn Sampler's recursive tree-doubling, and SVGD moving a set of particles to the posterior by a deterministic kernel flow. |
| `nn/_losses_extra4.py` | Soft-DTW: replacing DTW's hard `min` with a `-γ·logsumexp` soft-min so sequence alignment becomes differentiable. |
| `evolutionary/_advanced2.py` | NEAT: why growing a network's topology needs a topological-order forward pass (a new hidden node can outrank the output it feeds). |

## The API contract

`utils/estimator_checks.py` is worth reading as a *specification*: it enforces
the `fit`/`predict`/`transform` contract, parameter immutability in `__init__`,
the trailing-underscore convention for learned attributes, and `clone`
round-tripping. Reading it tells you exactly what "a nupyml estimator" means —
and it is the same check your own estimators can run against
(see [Contributing](contributing.md)).

## Performance context

Being readable does not mean being slow. Pure numpy/scipy means CPU-only,
single-threaded (no `n_jobs`), and no Cython — so scikit-learn keeps a
constant-factor edge on the heaviest estimators — but nothing is asymptotically
wrong, and a few estimators are actually faster. The README's performance table
has representative fit-time ratios.
