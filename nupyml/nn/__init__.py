"""Neural networks on the nupyml autograd engine."""
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
