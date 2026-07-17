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
  logits rather than probabilities.
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
    BCEWithLogitsLoss,
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
    "BCELoss", "BCEWithLogitsLoss",
    "SGD", "Adam", "AdamW", "RMSprop", "clip_grad_norm",
    "StepLR", "CosineAnnealingLR", "WarmupLR",
    "RNN", "GRU", "LSTM",
    "scaled_dot_product_attention", "MultiHeadAttention", "causal_mask",
    "TransformerEncoderLayer", "PositionalEncoding",
    "DataLoader", "MLPClassifier", "MLPRegressor", "init",
]
