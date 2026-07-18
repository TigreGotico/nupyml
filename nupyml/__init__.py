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
   layers built on it, up to the generative models (VAE, GAN, diffusion, flows)
   and what each trades away for what.
8. ``evolutionary`` -- optimization when there is no gradient at all, which
   throws the rest of the library's reliance on one into relief.
9. ``nonparametric`` -- flexible regression that stays readable: LOESS, GAM,
   MARS, RuleFit -- the ground between a linear model and a forest.
10. ``timeseries`` and ``sequence`` -- when ORDER is the signal: state-space
    filters, ARIMA, exponential smoothing, and the dynamic programs behind
    sequence labelling and sequence distance.
11. ``rl``, ``inference``, ``survival``, ``causal`` -- learning from
    consequences, sampling and honest uncertainty, time-to-event with censoring,
    and the effect of DOING rather than seeing.
12. ``search``, ``streaming``, ``patterns``, ``recommend`` -- scale: approximate
    nearest neighbours and sketches, online learners, association-rule mining,
    and matrix-factorization recommenders.
13. ``explain``, ``imbalance``, ``embed``, ``metric_learning`` -- interpreting a
    model after the fact, fixing class imbalance, learning word/subword
    representations, and learning the distance itself.
14. ``gnn`` -- graph neural networks: message passing on data that is a graph,
    where connectivity is the signal.
15. ``covariance``, ``cross_decomposition``, ``dummy`` -- shrinkage covariance
    estimation, latent structure between two variable blocks, and the baselines
    every real model must beat.
16. ``stats`` -- inference, not just prediction: standard errors, p-values,
    confidence intervals, and the diagnostic tests that say whether they are
    valid.
17. ``pgm`` -- probabilistic graphical models: Bayesian networks, Markov random
    fields, and exact inference by factor elimination.
18. ``anomaly`` and ``drift`` -- unsupervised anomaly scoring (histogram, CDF,
    angle, cluster based) and detecting when a data stream's distribution
    changes underneath a deployed model.
19. ``topic`` -- topic models (LDA, LSA), the BM25 search-ranking function, and
    document embeddings (Doc2Vec).
20. ``multilabel`` and ``active`` -- predicting a SET of labels per example
    (exploiting label correlations), and choosing which few examples are worth
    the cost of labelling.
21. ``optimal_transport`` -- the earth-mover's distance and its Sinkhorn solver,
    barycenters, domain adaptation, and the MMD/energy two-sample distances;
    plus ITML/LFDA/RCA in ``metric_learning``.
22. ``graph`` -- learning from connectivity: PageRank/HITS/centrality, community
    detection (Louvain, label propagation), node embeddings (DeepWalk, node2vec),
    and the Weisfeiler-Lehman graph kernel.
23. ``changepoint`` and ``copula`` -- finding where a signal's behaviour changes,
    and modelling the SHAPE of dependence between variables separately from their
    marginals; plus survival extensions (RSF, AFT, concordance).
24. ``encoders`` and the advanced selectors in ``feature_selection`` -- turning
    messy categoricals into numbers without leaking the target (WOE, James-Stein
    and other shrinkage encoders, leave-one-out), taming outliers/rare-levels/
    cycles, and all-relevant/redundancy-aware/stable feature selection (Boruta,
    mRMR, ReliefF, stability selection).
25. ``image`` -- the pre-deep-learning vision toolkit: HOG/LBP/GLCM/Gabor
    descriptors (each invariant to a chosen nuisance), the integral image for
    constant-time box sums, connected-component labelling, binary morphology,
    and the Hough line transform (finding lines by voting in parameter space).
26. balanced ensembles in ``imbalance`` (BalancedRandomForest, RUSBoost,
    EasyEnsemble), the conformal expansion in ``inference`` (conformalized
    quantile regression, Venn-Abers probability intervals, adaptive conformal),
    neighbourhood recommenders and SVD++ in ``recommend``, and the AutoML-lite
    searchers in ``model_selection`` (Hyperband, TPE).
27. ``tsclass`` -- time-series CLASSIFICATION (labelling a whole series by its
    shape): ROCKET random convolutional kernels, shapelet transform, BOSS
    symbolic bag-of-words, DTW k-NN, a tsfresh-style feature bank, and the matrix
    profile for motif/discord discovery.
28. ``fairness`` -- measuring and mitigating group bias: the conflicting
    group-fairness metrics (demographic parity, equal opportunity, equalized
    odds, predictive parity) and mitigators at each stage (reweighing and
    correlation removal, exponentiated-gradient reduction, threshold optimizing).
29. the interpretability expansion in ``explain``: the EBM/GA2M glass box (an
    additive model of plottable per-feature shapes), ALE plots (PDP's unbiased
    replacement under correlation), the H-statistic (interaction strength), global
    surrogate trees, anchor rules, and counterfactual explanations.
30. ``ordinal`` and its relatives -- predicting ORDERED categories
    (proportional-odds logit, rank-ridge), joint-support multi-output regression
    (``linear_model.MultiTaskElasticNet``), and conditional-quantile prediction
    (``ensemble.QuantileForest``) for calibrated intervals from a forest.
31. more ``cluster`` methods -- co-clustering rows and columns jointly
    (SpectralCoclustering), density-peak clustering (centres as dense points far
    from denser ones), and consensus clustering (a stable partition from many
    runs via a co-association matrix).
32. more ``anomaly`` detectors -- LODA (ensemble of random 1-D projection
    histograms), feature bagging (subspace ensemble), half-space trees (mass in
    random splits), Mahalanobis and PCA-reconstruction detectors -- plus automatic
    score thresholding (IQR, MAD, generalized ESD).
33. more ``graph`` learning -- link prediction (common-neighbours, Jaccard,
    Adamic-Adar, resource-allocation, preferential-attachment, Katz), degree
    assortativity, Girvan-Newman communities, max-flow/min-cut, shortest-path and
    random-walk kernels, and the LINE embedding.
34. the ``optimal_transport`` expansion -- unbalanced OT (mass-relaxed, robust to
    outliers), sliced Wasserstein (cheap high-dim via random 1-D projections),
    Gromov-Wasserstein (aligning distributions in DIFFERENT spaces), and the
    barycentric OT mapping for domain adaptation.
35. the ``survival`` expansion -- additive hazards (time-varying effects Cox
    forbids), competing risks (Aalen-Johansen cumulative incidence), the
    parametric Weibull AFT, and time-dependent evaluation (IPCW Brier score,
    integrated Brier, time-dependent AUC).
36. ``signal`` -- 1-D signal processing (the counterpart to ``image``): the
    discrete wavelet transform and denoising, STFT/spectrogram/mel/MFCC audio
    features, empirical mode decomposition, and detection tools (peaks, Hilbert
    envelope, zero-crossing rate).
37. ``ranking`` and more ``recommend`` -- learning to rank (RankNet's pairwise
    cross-entropy, LambdaMART's NDCG-scaled lambda gradients) with the ranking
    metrics that judge an order (NDCG, MAP, MRR, hit-rate), plus Slope One and
    co-clustering collaborative filtering.
38. AutoML and GP expansion -- BOHB search (``model_selection``, Hyperband's
    schedule with TPE's proposals), greedy ensemble selection (``ensemble``), and
    scalable/vector GPs (``gaussian_process``: inducing-point sparse GP and a
    multi-output GP).
39. more ``nn`` losses -- the long-tail / imbalance / count / contrastive zoo the
    base losses missed: class-balanced and balanced-softmax, PolyLoss,
    focal-Tversky, GHM (gradient-density harmonising), Wing (log-then-linear
    regression), Poisson NLL for counts, and supervised contrastive.
40. more ``nn`` optimizers & schedules -- Adadelta, AMSGrad, LAMB, and the
    wrappers SAM (flat minima), SWA and weight EMA; plus one-cycle, cyclical,
    polynomial, and exponential learning-rate schedules.
41. ``optimize`` -- from-scratch mathematical optimization under the estimators:
    linear programming (simplex), quadratic programming (active-set), proximal
    methods (FISTA, ADMM), conjugate gradient, and derivative-free search
    (Nelder-Mead, simulated annealing, particle swarm).
42. supervised categorical embeddings in ``embed`` -- a ``CategoricalVectorizer``
    for feature dicts, ``LabelGuidedEmbeddings`` (a label-trained MLP's hidden
    layer becomes the embedding), and ``EntityEmbeddingEncoder`` (learned dense
    vectors per high-cardinality category). Trained with autograd, run in numpy.
43. more ``nn`` architectures -- residual blocks, a transformer DECODER +
    GPT-style causal language model, a mixture density network (predict a
    multimodal distribution, not a point), and a weight-sharing Siamese network.
44. more self-supervision in ``nn`` -- Barlow Twins (redundancy reduction toward
    the identity cross-correlation) and BYOL (an EMA target + predictor, learning
    without negatives), joining the existing SimCLR and masked autoencoder.
45. the ``causal`` expansion -- instrumental variables (2SLS, for unobserved
    confounding), double ML (orthogonalised ATE with ML nuisances), T-/X-learners
    (heterogeneous effects), difference-in-differences and synthetic control
    (panel data), and NOTEARS (learning a causal DAG by continuous optimization).
46. approximate Bayesian inference in ``inference`` -- mean-field variational
    inference (ADVI, posterior by optimization), a sequential Monte Carlo sampler,
    approximate Bayesian computation (likelihood-free), and the GPLVM (nonlinear
    probabilistic PCA).
47. deep RL in ``rl`` -- DQN (a neural Q-function with experience replay and a
    target network) and PPO (the clipped-surrogate policy update), extending the
    tabular and policy-gradient methods to function approximation.
48. language & NLP -- an n-gram language model with Kneser-Ney smoothing,
    TextRank keyword/summary extraction, and SIF sentence embeddings (in
    ``topic``), plus GRU4Rec session-based recommendation (in ``recommend``).

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
* ``nn/vqvae.py``            -- the straight-through estimator: a deliberate lie
  about a derivative, and why it is the honest choice.
* ``evolutionary/strategies.py`` -- CMA-ES: a derivative-free quasi-Newton
  method, learning the landscape's shape into a covariance matrix.

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
