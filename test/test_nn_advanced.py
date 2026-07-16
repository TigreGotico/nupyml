import numpy as np
import pytest

from nupyml import nn
from nupyml.autograd import Tensor, gradcheck
from nupyml.autograd import functional as F

rng = np.random.RandomState(0)


# ---------------------------------------------------------------------------
# conv net end-to-end
# ---------------------------------------------------------------------------

def _toy_images(n=120, size=10, seed=0):
    """Class 0: horizontal bar; class 1: vertical bar (+ noise)."""
    r = np.random.RandomState(seed)
    X = r.normal(0, 0.2, size=(n, 1, size, size))
    y = r.randint(0, 2, size=n)
    for i in range(n):
        pos = r.randint(2, size - 2)
        if y[i] == 0:
            X[i, 0, pos, :] += 2.0
        else:
            X[i, 0, :, pos] += 2.0
    return X, y


def test_cnn_learns_bars():
    X, y = _toy_images(160)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1, rng=0), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1, rng=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(8 * 2 * 2, 2, rng=2),
    )
    opt = nn.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = nn.DataLoader(X, y, batch_size=32, random_state=0)
    for _ in range(15):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(Tensor(xb)), yb).backward()
            opt.step()
    model.eval()
    Xte, yte = _toy_images(60, seed=99)
    pred = np.argmax(model(Tensor(Xte)).data, axis=1)
    assert (pred == yte).mean() > 0.9


def test_batchnorm2d():
    bn = nn.BatchNorm2d(3)
    x = Tensor(rng.normal(5, 3, size=(8, 3, 4, 4)), requires_grad=True)
    out = bn(x)
    assert abs(out.data.mean()) < 0.1
    assert gradcheck(lambda a: bn(a), [x], rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# recurrent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell_cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_recurrent_shapes_and_grads(cell_cls):
    layer = cell_cls(input_size=5, hidden_size=7, rng=0)
    x = Tensor(rng.normal(size=(3, 4, 5)), requires_grad=True)
    outs, final = layer(x)
    assert outs.shape == (3, 4, 7)
    h = final[0] if isinstance(final, tuple) else final
    assert h.shape == (3, 7)
    assert gradcheck(lambda a: layer(a)[0], [x], eps=1e-5, rtol=1e-3, atol=1e-5)


def _parity_sequences(n, T, seed=0):
    """y = parity of the number of 1s in a binary sequence."""
    r = np.random.RandomState(seed)
    X = r.randint(0, 2, size=(n, T, 1)).astype(float)
    y = (X.sum(axis=(1, 2)) % 2).astype(int)
    return X, y


def test_lstm_learns_parity():
    X, y = _parity_sequences(300, 8)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 16, rng=0)
            self.head = nn.Linear(16, 2, rng=1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h)

    model = Model()
    opt = nn.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = nn.DataLoader(X, y, batch_size=32, random_state=0)
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(Tensor(xb)), yb).backward()
            opt.step()
    Xte, yte = _parity_sequences(100, 8, seed=9)
    pred = np.argmax(model(Tensor(Xte)).data, axis=1)
    assert (pred == yte).mean() > 0.85


# ---------------------------------------------------------------------------
# attention / transformer
# ---------------------------------------------------------------------------

def test_attention_shapes_and_softmax():
    q = Tensor(rng.normal(size=(2, 3, 5, 8)))
    out, attn = nn.scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 3, 5, 8)
    assert np.allclose(attn.data.sum(axis=-1), 1)


def test_causal_mask_blocks_future():
    T = 4
    q = Tensor(rng.normal(size=(1, 1, T, 8)))
    _, attn = nn.scaled_dot_product_attention(q, q, q, mask=nn.causal_mask(T))
    assert np.allclose(np.triu(attn.data[0, 0], k=1), 0)


def test_multihead_attention_grad():
    mha = nn.MultiHeadAttention(embed_dim=8, num_heads=2, rng=0)
    x = Tensor(rng.normal(size=(2, 3, 8)), requires_grad=True)
    out = mha(x)
    assert out.shape == (2, 3, 8)
    assert gradcheck(lambda a: mha(a), [x], eps=1e-5, rtol=1e-3, atol=1e-5)


def test_transformer_layer():
    layer = nn.TransformerEncoderLayer(embed_dim=8, num_heads=2, rng=0)
    x = Tensor(rng.normal(size=(2, 5, 8)), requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 5, 8)


def test_positional_encoding():
    pe = nn.PositionalEncoding(8, max_len=100)
    x = Tensor(np.zeros((1, 10, 8)))
    out = pe(x).data
    assert out.shape == (1, 10, 8)
    assert not np.allclose(out[0, 0], out[0, 5])


def test_tiny_transformer_overfits_copy_task():
    """Char-level: predict the next token of a fixed repeating sequence."""
    vocab, T, D = 5, 8, 32
    r = np.random.RandomState(0)
    base = r.randint(0, vocab, size=200)
    X = np.stack([base[i:i + T] for i in range(150)])
    Y = np.stack([base[i + 1:i + T + 1] for i in range(150)])

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, D, rng=0)
            self.pos = nn.PositionalEncoding(D)
            self.block = nn.TransformerEncoderLayer(D, num_heads=2, rng=1)
            self.head = nn.Linear(D, vocab, rng=2)

        def forward(self, idx):
            x = self.pos(self.emb(idx))
            x = self.block(x, mask=nn.causal_mask(idx.shape[1]))
            return self.head(x)

    model = TinyGPT()
    opt = nn.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(300):
        opt.zero_grad()
        logits = model(X)                       # (n, T, vocab)
        loss = loss_fn(logits.reshape(-1, vocab), Y.ravel())
        loss.backward()
        opt.step()
    logits = model(X)
    pred = np.argmax(logits.data, axis=-1)
    # early positions have too little context to be predictable; score the
    # second half of each window
    acc = (pred[:, 4:] == Y[:, 4:]).mean()
    assert acc > 0.85, acc
