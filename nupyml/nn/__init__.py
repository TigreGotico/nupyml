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
from .autoencoder import (
    AutoEncoder, DenoisingAutoEncoder, SparseAutoEncoder, VAE, ConditionalVAE,
    VAELoss, kl_divergence_normal,
)
from .gan import (
    Generator, Discriminator, GANLoss, NonSaturatingGANLoss, WGANLoss, GAN,
)
from .optim import (
    SGD, Adam, AdamW, RMSprop, clip_grad_norm, StepLR, CosineAnnealingLR,
    WarmupLR,
)
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
    "SGD", "Adam", "AdamW", "RMSprop", "clip_grad_norm",
    "StepLR", "CosineAnnealingLR", "WarmupLR",
    "RNN", "GRU", "LSTM",
    "scaled_dot_product_attention", "MultiHeadAttention", "causal_mask",
    "TransformerEncoderLayer", "PositionalEncoding",
    "DataLoader", "MLPClassifier", "MLPRegressor", "init",
]
