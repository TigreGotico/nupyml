"""Session-based sequential recommendation with a GRU (GRU4Rec).

Classic collaborative filtering models a user's static taste. Many settings have
no persistent user at all -- an anonymous browsing SESSION -- where what matters
is the ORDER of the last few clicks. GRU4Rec models the session as a sequence and
predicts the next item from the running hidden state.
"""
import numpy as np

from ..base import BaseEstimator
from ..autograd import Tensor
from .. import nn


class GRU4Rec(BaseEstimator):
    """Predict the next item in a session with a GRU (Hidasi et al., 2016).

    THE SHIFT FROM MATRIX FACTORIZATION
    -----------------------------------
    Factorization asks "what does this USER like overall?". GRU4Rec asks "given the
    sequence of items clicked SO FAR this session, what comes next?" -- there may be
    no user id at all. It embeds each item, runs a GRU over the session so the
    hidden state summarises the trajectory, and projects that state to a score per
    item. Because the recommendation depends on the ORDER and RECENCY of clicks
    (not a static profile), it captures intent that drifts within a visit -- the
    reason session models beat CF on e-commerce and media browsing.

    ``fit`` takes sessions (lists of item ids); ``recommend`` returns the top-k
    next items for a session prefix.
    """

    def __init__(self, n_items, embed_dim=32, hidden=64, epochs=20, lr=0.01,
                 random_state=None):
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def _build(self, rng):
        self.embed_ = nn.Embedding(self.n_items, self.embed_dim, rng=rng)
        self.gru_ = nn.GRU(self.embed_dim, self.hidden, rng=rng)
        self.out_ = nn.Linear(self.hidden, self.n_items, rng=rng)

    def _params(self):
        return (list(self.embed_.parameters()) + list(self.gru_.parameters())
                + list(self.out_.parameters()))

    def _logits(self, session):
        x = self.embed_(np.asarray(session)[None])    # (1, T, embed)
        seq, _ = self.gru_(x)                          # (1, T, hidden)
        return self.out_(seq)[0]                        # (T, n_items)

    def fit(self, sessions):
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        self._build(rng)
        opt = nn.Adam(self._params(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        sessions = [list(s) for s in sessions if len(s) >= 2]
        for _ in range(self.epochs):
            rng.shuffle(sessions)
            for s in sessions:
                # predict item t+1 from the state after item t (teacher forcing)
                logits = self._logits(s[:-1])          # (T-1, n_items)
                targets = np.asarray(s[1:])
                opt.zero_grad()
                loss_fn(logits, targets).backward()
                opt.step()
        return self

    def recommend(self, session, k=5, exclude_seen=True):
        logits = self._logits(list(session)).data[-1]  # scores after the last item
        if exclude_seen:
            logits = logits.copy()
            logits[list(session)] = -np.inf
        return np.argsort(logits)[::-1][:k]

    def predict_next(self, session):
        return int(self.recommend(session, k=1, exclude_seen=False)[0])


__all__ = ["GRU4Rec"]
