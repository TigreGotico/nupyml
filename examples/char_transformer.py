"""Overfit a tiny causal transformer on a repeating character sequence."""
import numpy as np

from nupyml import nn
from nupyml.autograd import Tensor

text = "hello world! " * 40
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text])

T, D, V = 16, 32, len(chars)
X = np.stack([data[i:i + T] for i in range(len(data) - T - 1)])
Y = np.stack([data[i + 1:i + T + 1] for i in range(len(data) - T - 1)])


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D, rng=0)
        self.pos = nn.PositionalEncoding(D)
        self.block = nn.TransformerEncoderLayer(D, num_heads=4, rng=1)
        self.head = nn.Linear(D, V, rng=2)

    def forward(self, idx):
        x = self.pos(self.emb(idx))
        x = self.block(x, mask=nn.causal_mask(idx.shape[1]))
        return self.head(x)


model = TinyGPT()
opt = nn.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for step in range(200):
    batch = np.random.RandomState(step).choice(len(X), size=64)
    opt.zero_grad()
    loss = loss_fn(model(X[batch]).reshape(-1, V), Y[batch].ravel())
    loss.backward()
    opt.step()
    if step % 40 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# greedy generation
ctx = list(data[:T])
for _ in range(40):
    logits = model(np.array([ctx[-T:]]))
    ctx.append(int(np.argmax(logits.data[0, -1])))
print("generated:", "".join(chars[i] for i in ctx[T:]))
