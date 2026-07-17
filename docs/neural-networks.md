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
  `Flatten`.
- **Normalization**: `BatchNorm1d` / `BatchNorm2d`, `LayerNorm`.
- **Regularization**: `Dropout`.
- **Sequence**: `Embedding`, `RNN`, `GRU`, `LSTM`.
- **Attention**: `MultiHeadAttention`, `TransformerEncoderLayer`,
  `PositionalEncoding`.
- **Containers**: `Sequential`, and any subclass of `Module`.

`autograd/functional.py` shows convolution implemented as a matmul (im2col) and
the numerically stable softmax / log-softmax.

## Losses and optimizers

- **Losses**: MSE, MAE, Huber, cross-entropy (with a fused log-softmax for
  stability), BCE, BCE-with-logits.
- **Optimizers**: SGD (with momentum / Nesterov), Adam, AdamW, RMSprop.
- **Schedulers**: StepLR, cosine, warmup.
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
