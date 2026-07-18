"""Neural networks: composable layers over the autograd engine.

Everything here is built from ``nupyml.autograd`` -- no layer computes its own
gradient, because it does not have to. Define the forward pass with Tensor
operations and the backward pass follows from the tape. That is the whole payoff
of writing an autodiff engine, and it is why adding a new layer here means
writing one ``forward`` method.

THE PIECES
----------
* ``module.py``    -- ``Module``, the base class that tracks parameters so
  optimizers and ``state_dict`` can find them.
* ``layers.py``    -- Linear, Conv2d, normalization, dropout, embeddings.
* ``recurrent.py`` -- RNN, GRU, LSTM, and the vanishing-gradient problem gates
  exist to solve.
* ``attention.py`` -- attention and transformer blocks.
* ``losses.py``    -- what "wrong" means, and why the classification losses take
  logits rather than probabilities. Focal, label smoothing, Dice/Tversky,
  distillation, Wasserstein.
* ``metric_losses.py`` -- losses that shape an EMBEDDING rather than a boundary:
  contrastive, triplet, InfoNCE, ArcFace. How self-supervised learning works.
* ``ctc.py``       -- training a sequence model with no alignment, by summing
  over all of them.
* ``autoencoder.py`` -- AE, denoising, sparse, and the **VAE**: the
  reparameterization trick and the ELBO, which is where a compression model
  becomes a generative one.
* ``gan.py``       -- adversarial training, why the obvious generator loss
  saturates, and what the Wasserstein critic fixes.
* ``vqvae.py``     -- a DISCRETE latent, and the straight-through estimator that
  gets a gradient through an argmin.
* ``flows.py``     -- RealNVP: an EXACT likelihood, bought by making the map
  invertible and its Jacobian triangular.
* ``diffusion.py`` -- DDPM: destroy the data gradually, learn to undo one step.
* ``autoregressive.py`` -- MADE: an exact likelihood from the chain rule, and
  masks that make the ordering structural rather than a convention.
* ``rbm.py``       -- an energy-based model, and contrastive divergence: what to
  do when the normalising constant cannot be computed.
* ``self_supervised.py`` -- SimCLR and masked autoencoding: inventing labels from
  the data, and why the pretext task is the method.

The generative modules are best read in that order. Each one gives up something
the previous kept -- a tractable likelihood, a continuous latent, a single-shot
decoder -- and gets something back for it. Comparing what each trades away is
more instructive than any of them alone.
* ``optim.py``     -- SGD through Adam, as a progression of ideas.
* ``mlp.py``       -- ``MLPClassifier``/``MLPRegressor``, sklearn-style wrappers
  for when you want a network without writing a training loop.

TRAIN AND EVAL ARE DIFFERENT
----------------------------
``Dropout`` and ``BatchNorm`` behave differently during training and inference.
Call ``model.eval()`` before predicting and ``model.train()`` before training.
Forgetting is the most common source of "my model works in training and not
after".
"""
from .module import Parameter, Module, Sequential, ModuleList
from .layers import (
    Linear, ReLU, LeakyReLU, GELU, Tanh, Sigmoid, Softmax, Flatten, Dropout,
    Embedding, Conv2d, MaxPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, LayerNorm,
)
from .losses import (
    MSELoss, MAELoss, HuberLoss, CrossEntropyLoss, NLLLoss, BCELoss,
    BCEWithLogitsLoss, FocalLoss, LabelSmoothingCrossEntropy, KLDivLoss,
    DistillationLoss, HingeEmbeddingLoss, MarginRankingLoss, DiceLoss,
    TverskyLoss, IoULoss, LogCoshLoss, SmoothL1Loss, QuantileLoss,
    TweedieLoss, WassersteinLoss, gradient_penalty,
)
from .metric_losses import (
    ContrastiveLoss, TripletLoss, BatchHardTripletLoss, NPairsLoss,
    InfoNCELoss, CosFaceLoss, ArcFaceLoss, CenterLoss,
)
from .ctc import CTCLoss, ctc_greedy_decode
from ._losses_extra import (
    ClassBalancedLoss, BalancedSoftmaxLoss, PolyLoss, FocalTverskyLoss,
    GHMLoss, WingLoss, PoissonNLLLoss, SupConLoss,
)
from ._optim_extra import (
    Adadelta, AMSGrad, LAMB, SAM, SWA, EMA,
    ExponentialLR, PolynomialLR, CyclicalLR, OneCycleLR,
)
from ._architectures import (
    ResidualBlock, TransformerDecoderLayer, CausalLanguageModel,
    MixtureDensityNetwork, SiameseNetwork,
)
from ._self_sup_extra import BarlowTwins, BYOL
from ._compression import (magnitude_prune, quantize_weights,
                           low_rank_approximation, KnowledgeDistillation)
from ._architectures2 import (DeepSets, SetTransformer, GIN, LinearAttention,
                              NeuralODE, StateSpaceModel, KAN)
from ._optim_extra2 import (AdaBelief, Adamax, Yogi, AdaBound, Adafactor,
                            pcgrad, NaturalGradient)
from ._losses_extra2 import (SymmetricCrossEntropy, GeneralizedCrossEntropy,
                            AsymmetricLoss, CharbonnierLoss, MSLELoss, SMAPELoss,
                            ListMLELoss, VICRegLoss)
from ._flows2 import MAF
from ._architectures3 import (HighwayNetwork, SqueezeExcitation, CausalConv1d,
                             TemporalConvNet, GATv2, HyperNetwork)
from ._optim_extra3 import Shampoo, LARS, LARC, NovoGrad, Adan
from ._losses_extra3 import (CircleLoss, MultiSimilarityLoss, AngularLoss,
                             BoxRegressionLoss, SSIMLoss, TukeyBiweightLoss)
from .autoencoder import (
    AutoEncoder, DenoisingAutoEncoder, SparseAutoEncoder, VAE, ConditionalVAE,
    VAELoss, kl_divergence_normal,
)
from .gan import (
    Generator, Discriminator, GANLoss, NonSaturatingGANLoss, WGANLoss, GAN,
)
from .vqvae import VectorQuantizer, VQVAE
from .flows import AffineCoupling, RealNVP
from .diffusion import (
    DDPM, NoisePredictor, linear_beta_schedule, cosine_beta_schedule,
    timestep_embedding,
)
from .rbm import BernoulliRBM
from .autoregressive import MADE, MaskedLinear
from .self_supervised import (
    SimCLR, MaskedAutoEncoder, GaussianNoiseAugment, MaskingAugment,
)
from .optim import (
    SGD, Adam, AdamW, RMSprop, clip_grad_norm, StepLR, CosineAnnealingLR,
    WarmupLR,
)
from .activations import SiLU, Mish, ELU, SELU, Softplus, GLU, SwiGLU
from .normalization import RMSNorm, GroupNorm, InstanceNorm
from .optim_extra import Adagrad, Nadam, RAdam, Lion, Lookahead
from .positional import RotaryPositionalEmbedding, ALiBi
from .regularization import mixup, cutmix, StochasticDepth, drop_path
from .recurrent import RNN, GRU, LSTM
from .attention import (
    scaled_dot_product_attention, MultiHeadAttention, causal_mask,
    TransformerEncoderLayer, PositionalEncoding,
)
from .data import DataLoader
from .mlp import MLPClassifier, MLPRegressor
from . import init

__all__ = [
    "Parameter", "Module", "Sequential", "ModuleList",
    "Linear", "ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "Softmax",
    "Flatten", "Dropout", "Embedding", "Conv2d", "MaxPool2d", "AvgPool2d",
    "BatchNorm1d", "BatchNorm2d", "LayerNorm",
    "MSELoss", "MAELoss", "HuberLoss", "CrossEntropyLoss", "NLLLoss",
    "BCELoss", "BCEWithLogitsLoss", "FocalLoss", "LabelSmoothingCrossEntropy",
    "KLDivLoss", "DistillationLoss", "HingeEmbeddingLoss", "MarginRankingLoss",
    "DiceLoss", "TverskyLoss", "IoULoss", "LogCoshLoss", "SmoothL1Loss",
    "QuantileLoss", "TweedieLoss", "WassersteinLoss", "gradient_penalty",
    "ContrastiveLoss", "TripletLoss", "BatchHardTripletLoss", "NPairsLoss",
    "InfoNCELoss", "CosFaceLoss", "ArcFaceLoss", "CenterLoss",
    "CTCLoss", "ctc_greedy_decode",
    "AutoEncoder", "DenoisingAutoEncoder", "SparseAutoEncoder", "VAE",
    "ConditionalVAE", "VAELoss", "kl_divergence_normal",
    "Generator", "Discriminator", "GANLoss", "NonSaturatingGANLoss",
    "WGANLoss", "GAN",
    "VectorQuantizer", "VQVAE", "AffineCoupling", "RealNVP",
    "DDPM", "NoisePredictor", "linear_beta_schedule", "cosine_beta_schedule",
    "timestep_embedding", "BernoulliRBM", "MADE", "MaskedLinear",
    "SimCLR", "MaskedAutoEncoder", "GaussianNoiseAugment", "MaskingAugment",
    "SGD", "Adam", "AdamW", "RMSprop", "clip_grad_norm",
    "StepLR", "CosineAnnealingLR", "WarmupLR",
    "SiLU", "Mish", "ELU", "SELU", "Softplus", "GLU", "SwiGLU",
    "RMSNorm", "GroupNorm", "InstanceNorm",
    "Adagrad", "Nadam", "RAdam", "Lion", "Lookahead",
    "RotaryPositionalEmbedding", "ALiBi",
    "mixup", "cutmix", "StochasticDepth", "drop_path",
    "RNN", "GRU", "LSTM",
    "scaled_dot_product_attention", "MultiHeadAttention", "causal_mask",
    "TransformerEncoderLayer", "PositionalEncoding",
    "DataLoader", "MLPClassifier", "MLPRegressor", "init",
    "ClassBalancedLoss", "BalancedSoftmaxLoss", "PolyLoss", "FocalTverskyLoss",
    "GHMLoss", "WingLoss", "PoissonNLLLoss", "SupConLoss",
    "Adadelta", "AMSGrad", "LAMB", "SAM", "SWA", "EMA",
    "ExponentialLR", "PolynomialLR", "CyclicalLR", "OneCycleLR",
    "ResidualBlock", "TransformerDecoderLayer", "CausalLanguageModel",
    "MixtureDensityNetwork", "SiameseNetwork",
    "BarlowTwins", "BYOL",
    "magnitude_prune", "quantize_weights", "low_rank_approximation",
    "KnowledgeDistillation",
    "DeepSets", "SetTransformer", "GIN", "LinearAttention", "NeuralODE",
    "StateSpaceModel", "KAN",
    "AdaBelief", "Adamax", "Yogi", "AdaBound", "Adafactor", "pcgrad",
    "NaturalGradient",
    "SymmetricCrossEntropy", "GeneralizedCrossEntropy", "AsymmetricLoss",
    "CharbonnierLoss", "MSLELoss", "SMAPELoss", "ListMLELoss", "VICRegLoss",
    "MAF",
    "HighwayNetwork", "SqueezeExcitation", "CausalConv1d", "TemporalConvNet",
    "GATv2", "HyperNetwork",
    "Shampoo", "LARS", "LARC", "NovoGrad", "Adan",
    "CircleLoss", "MultiSimilarityLoss", "AngularLoss", "BoxRegressionLoss",
    "SSIMLoss", "TukeyBiweightLoss",
]
