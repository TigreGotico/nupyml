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
from .class_balanced_loss import ClassBalancedLoss
from .balanced_softmax_loss import BalancedSoftmaxLoss
from .poly_loss import PolyLoss
from .focal_tversky_loss import FocalTverskyLoss
from .ghm_loss import GHMLoss
from .wing_loss import WingLoss
from .poisson_nll_loss import PoissonNLLLoss
from .sup_con_loss import SupConLoss
from .adadelta import Adadelta
from .ams_grad import AMSGrad
from .lamb import LAMB
from .sam import SAM
from .swa import SWA
from .ema import EMA
from .exponential_lr import ExponentialLR
from .polynomial_lr import PolynomialLR
from .cyclical_lr import CyclicalLR
from .one_cycle_lr import OneCycleLR
from .residual_block import ResidualBlock
from .transformer_decoder_layer import TransformerDecoderLayer
from .causal_language_model import CausalLanguageModel
from .mixture_density_network import MixtureDensityNetwork
from .siamese_network import SiameseNetwork
from ._self_sup_extra import BarlowTwins, BYOL
from ._compression import (magnitude_prune, quantize_weights,
                           low_rank_approximation, KnowledgeDistillation)
from .deep_sets import DeepSets
from .set_transformer import SetTransformer
from .gin import GIN
from .linear_attention import LinearAttention
from .neural_ode import NeuralODE
from .state_space_model import StateSpaceModel
from .kan import KAN
from .ada_belief import AdaBelief
from .adamax import Adamax
from .yogi import Yogi
from .ada_bound import AdaBound
from .adafactor import Adafactor
from .pcgrad import pcgrad
from .natural_gradient import NaturalGradient
from .symmetric_cross_entropy import SymmetricCrossEntropy
from .generalized_cross_entropy import GeneralizedCrossEntropy
from .asymmetric_loss import AsymmetricLoss
from .charbonnier_loss import CharbonnierLoss
from .msle_loss import MSLELoss
from .smape_loss import SMAPELoss
from .list_mle_loss import ListMLELoss
from .vic_reg_loss import VICRegLoss
from .maf import MAF
from .highway_network import HighwayNetwork
from .squeeze_excitation import SqueezeExcitation
from .causal_conv1d import CausalConv1d
from .temporal_conv_net import TemporalConvNet
from .ga_tv2 import GATv2
from .hyper_network import HyperNetwork
from .shampoo import Shampoo
from .lars_larc import (LARS, LARC)
from .novo_grad import NovoGrad
from .adan import Adan
from .circle_loss import CircleLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .angular_loss import AngularLoss
from .box_regression_loss import BoxRegressionLoss
from .ssim_loss import SSIMLoss
from .tukey_biweight_loss import TukeyBiweightLoss
from .wave_net import WaveNet
from .capsule_layer import CapsuleLayer
from .spatial_transformer import SpatialTransformer
from .echo_state_network import EchoStateNetwork
from .pointer_network import PointerNetwork
from .soft_dtw import soft_dtw
from .evidential_loss import EvidentialLoss
from .seesaw_loss import SeesawLoss
from .rank_net_loss import RankNetLoss
from .boundary_loss import BoundaryLoss
from ._dbn import DeepBeliefNetwork
from .vision_transformer import VisionTransformer
from .neural_turing_machine import NeuralTuringMachine
from .glow import Glow
from .ddim_sample import ddim_sample
from .relative_position_attention import RelativePositionAttention
from .barron_loss import BarronLoss
from .lovasz_hinge import lovasz_hinge
from .crps_ensemble import crps_ensemble
from .list_net_loss import ListNetLoss
from .proxy_anchor_loss import ProxyAnchorLoss
from .bidirectional import Bidirectional
from .differentiable_neural_computer import DifferentiableNeuralComputer
from .neural_spline_flow import NeuralSplineFlow
from .modern_hopfield_network import ModernHopfieldNetwork
from .graph_transformer import GraphTransformer
from .energy_score import energy_score
from .dino_loss import DINOLoss
from .distribution_focal_loss import distribution_focal_loss
from .region_mutual_information import region_mutual_information
from .box_regression_loss2 import BoxRegressionLoss2
from .conditional_neural_process import ConditionalNeuralProcess
from .pointnet import PointNet
from .energy_based_model import EnergyBasedModel
from .simsiam import SimSiam
from .vgae import VGAE
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
from .adagrad import Adagrad
from .nadam import Nadam
from .r_adam import RAdam
from .lion import Lion
from .lookahead import Lookahead
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
    "WaveNet", "CapsuleLayer", "SpatialTransformer", "EchoStateNetwork",
    "PointerNetwork",
    "soft_dtw", "EvidentialLoss", "SeesawLoss", "RankNetLoss", "BoundaryLoss",
    "DeepBeliefNetwork",
    "VisionTransformer", "NeuralTuringMachine", "Glow", "ddim_sample",
    "RelativePositionAttention",
    "BarronLoss", "lovasz_hinge", "crps_ensemble", "ListNetLoss",
    "ProxyAnchorLoss",
    "Bidirectional", "DifferentiableNeuralComputer", "NeuralSplineFlow",
    "ModernHopfieldNetwork", "GraphTransformer",
    "energy_score", "DINOLoss", "distribution_focal_loss",
    "region_mutual_information", "BoxRegressionLoss2",
    "ConditionalNeuralProcess", "PointNet", "EnergyBasedModel", "SimSiam", "VGAE",
]
