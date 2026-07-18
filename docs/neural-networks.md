# Neural networks

The neural stack is a small, PyTorch-shaped API built on a tape-based
reverse-mode autograd engine. It lives in `nupyml.autograd` (the engine) and
`nupyml.nn` (layers, losses, optimizers, and the estimator wrappers).

## Autograd

`nupyml.autograd.Tensor` wraps a numpy array and records the operations applied
to it on a tape. Calling `.backward()` walks that tape in reverse topological
order, accumulating gradients into every tensor that requires them. Every op is
gradient-checked against finite differences in the test suite, so the engine is
trustworthy to build on.

```python
from nupyml.autograd import Tensor

x = Tensor([[1.0, 2.0]], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)        # [[2., 4.]]
```

`autograd/tensor.py` is ~400 lines and is one of the best files to read first:
it explains why reverse mode (not forward), why a tape, and why topological
order matters.

## Layers

`nupyml.nn` provides the standard building blocks, all as autograd `Module`s:

- **Dense / conv**: `Linear`, `Conv2d` (im2col), `MaxPool2d` / `AvgPool2d`,
  `CausalConv1d`, `Flatten`.
- **Normalization**: `BatchNorm1d` / `BatchNorm2d`, `LayerNorm`, `RMSNorm`,
  `GroupNorm`, `InstanceNorm`.
- **Activations / gating**: `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Mish`, `ELU`,
  `SELU`, `Softplus`, `GLU`, `SwiGLU`.
- **Regularization**: `Dropout`, `StochasticDepth`.
- **Sequence**: `Embedding`, `RNN`, `GRU`, `LSTM`.
- **Attention**: `MultiHeadAttention`, `TransformerEncoderLayer` /
  `TransformerDecoderLayer`, `PositionalEncoding`, rotary embeddings, ALiBi,
  linear/Performer attention.
- **Containers**: `Sequential`, `ModuleList`, and any subclass of `Module`.

### Architectures

Beyond the primitives, complete architectures are provided as modules:

- **Sets / graphs**: `DeepSets`, `SetTransformer`, `GIN`, `GATv2`.
- **Sequence models**: `TemporalConvNet` (dilated causal), `WaveNet` (gated
  dilated-causal), `StateSpaceModel` (S4), `PointerNetwork`, `EchoStateNetwork`
  (reservoir computing), `CausalLanguageModel`.
- **Deep-net building blocks**: `ResidualBlock`, `HighwayNetwork`,
  `SqueezeExcitation`, `HyperNetwork` (a net that generates another's weights),
  `SpatialTransformer`, `CapsuleLayer` (dynamic routing), `KAN`, `NeuralODE`,
  `MixtureDensityNetwork`, `SiameseNetwork`, `DeepBeliefNetwork` (stacked RBMs),
  `VisionTransformer` (patch tokens), `NeuralTuringMachine` /
  `DifferentiableNeuralComputer` (addressable external memory), `Bidirectional`
  (BiLSTM/BiGRU), `ModernHopfieldNetwork` (attention as associative memory),
  `GraphTransformer` (edge-feature attention), `RelativePositionAttention`.
- **Generative**: `AutoEncoder` (+ denoising / sparse), `VAE` / `ConditionalVAE`,
  `VQVAE`, `GAN` / `WGAN`, `DDPM` diffusion (with `ddim_sample` for fast
  deterministic sampling), normalizing flows (`RealNVP`, `MAF`, `Glow`).
- **Self-supervised**: `SimCLR`, `BYOL`, `BarlowTwins`, `MaskedAutoEncoder`,
  plus `KnowledgeDistillation`.

`autograd/functional.py` shows convolution implemented as a matmul (im2col) and
the numerically stable softmax / log-softmax.

## Losses, optimizers, schedulers

- **Losses** (40+): the base set — MSE, MAE, Huber, cross-entropy (fused
  log-softmax), BCE, BCE-with-logits — plus:
  - *segmentation*: Dice, Tversky, focal-Tversky, IoU, boundary;
  - *imbalance / robustness*: focal, class-balanced, PolyLoss, GHM, seesaw,
    symmetric/generalized/asymmetric cross-entropy, Tukey biweight;
  - *metric learning*: contrastive, triplet, N-pairs, InfoNCE, SupCon, ArcFace,
    CosFace, center, Circle, multi-similarity, angular;
  - *ranking*: RankNet (pairwise), ListNet (listwise);
  - *robust / probabilistic*: Barron adaptive (spans L2/Charbonnier/Cauchy/Welsch),
    Tukey biweight, CRPS and the multivariate energy score (proper scores),
    distribution focal loss;
  - *segmentation*: Dice, Tversky, boundary, Lovász-softmax (IoU surrogate),
    region mutual information; *detection*: IoU/GIoU/DIoU/CIoU/EIoU/SIoU;
  - *metric / self-supervised*: proxy-anchor, ArcFace/CosFace/SupCon, DINO;
  - *other*: soft-DTW (differentiable alignment), evidential (Dirichlet
    uncertainty), SSIM, VICReg, quantile, Tweedie, Wasserstein.
- **Optimizers**: SGD (momentum / Nesterov), Adam, AdamW, RMSprop, Adagrad,
  Adadelta, Nadam, RAdam, Adamax, AMSGrad, Lion, Lookahead, AdaBelief, Yogi,
  AdaBound, Adafactor, LAMB, SAM, SWA, EMA, natural gradient, and **Shampoo,
  LARS / LARC, NovoGrad, Adan**.
- **Schedulers**: StepLR, cosine annealing, warmup, exponential, polynomial,
  cyclical, one-cycle.
- **Utilities**: gradient clipping, `DataLoader` for minibatching.

## Estimator wrappers

If you want the scikit-learn API for a network, `nn.MLPClassifier` and
`nn.MLPRegressor` wrap a multilayer perceptron behind `fit` / `predict`, with
`adam`, `sgd`, and `lbfgs` solvers, `early_stopping`, and `warm_start`. They pass
`check_estimator` and drop into pipelines like any other estimator.

## Persistence

Any model persists to a single npz file:

```python
model.save("weights.npz")
model.load("weights.npz")
```

## Worth reading

`nn/vqvae.py` implements the straight-through estimator — a deliberate lie about
a derivative, and a good illustration of when that is the honest engineering
choice. See the [reading guide](reading-guide.md) for more.
