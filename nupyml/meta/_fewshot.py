"""Few-shot classifiers: recognise a new class from a handful of examples.

Both learn an EMBEDDING and then classify by comparison, never by a fixed output
layer -- which is what lets them handle classes unseen at training time. They
differ only in how they compare a query to the labelled "support" examples:
prototypical networks compare to each class's MEAN embedding; matching networks
take an attention-weighted vote over EVERY support example.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..nn import Linear
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class _Embedder:
    """A small 2-layer embedding network shared by both few-shot models."""

    def __init__(self, in_dim, hidden, embed_dim, rng):
        self.l1 = Linear(in_dim, hidden, rng=rng)
        self.l2 = Linear(hidden, embed_dim, rng=rng)

    def __call__(self, x):
        return self.l2(self.l1(Tensor._wrap(x)).relu())

    def params(self):
        return list(self.l1.parameters()) + list(self.l2.parameters())


def _sample_episode(X, y, classes, n_way, k_shot, n_query, rng):
    chosen = rng.choice(classes, n_way, replace=False)
    sx, sy, qx, qy = [], [], [], []
    for lbl, c in enumerate(chosen):
        idx = rng.permutation(np.where(y == c)[0])[:k_shot + n_query]
        sx.append(X[idx[:k_shot]]); sy += [lbl] * k_shot
        qx.append(X[idx[k_shot:k_shot + n_query]]); qy += [lbl] * (len(idx) - k_shot)
    return (np.vstack(sx), np.array(sy),
            np.vstack(qx), np.array(qy), n_way)


class PrototypicalNetwork(BaseEstimator):
    """Classify a query by its nearest class PROTOTYPE (Snell et al., 2017).

    Embed the few labelled "support" examples, average each class's embeddings into
    a single PROTOTYPE, and label a query by whichever prototype it is closest to
    (negative squared distance as logits). Training runs in EPISODES that mimic the
    few-shot test: each episode picks a few classes and a few shots, so the network
    learns an embedding where class means are good classifiers -- which is exactly
    the condition met at test time on brand-new classes.
    """

    def __init__(self, hidden=32, embed_dim=16, n_way=3, k_shot=5, n_query=5,
                 episodes=300, lr=0.01, random_state=None):
        self.hidden = hidden; self.embed_dim = embed_dim
        self.n_way = n_way; self.k_shot = k_shot; self.n_query = n_query
        self.episodes = episodes; self.lr = lr; self.random_state = random_state

    def _prototypes(self, support_emb, sy, n_way):
        protos = [support_emb[np.where(sy == c)[0]].mean(axis=0)
                  for c in range(n_way)]
        return _stack(protos)

    def _logits(self, query_emb, protos):
        # -||q - p||^2 as logits (broadcast query x prototype)
        q2 = (query_emb * query_emb).sum(axis=1, keepdims=True)
        p2 = (protos * protos).sum(axis=1, keepdims=True).transpose()
        cross = query_emb @ protos.transpose()
        return -(q2 - 2.0 * cross + p2)

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        rng = check_random_state(self.random_state)
        classes = np.unique(y)
        self.embedder_ = _Embedder(X.shape[1], self.hidden, self.embed_dim, rng)
        params = self.embedder_.params()
        for _ in range(self.episodes):
            sx, sy, qx, qy, n_way = _sample_episode(
                X, y, classes, self.n_way, self.k_shot, self.n_query, rng)
            se = self.embedder_(sx)
            qe = self.embedder_(qx)
            protos = self._prototypes(se, sy, n_way)
            logits = self._logits(qe, protos)
            loss = _cross_entropy(logits, qy)
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def predict(self, support_X, support_y, query_X):
        """Label ``query_X`` from a fresh labelled support set."""
        support_y = np.asarray(support_y)
        classes = np.unique(support_y)
        remap = {c: i for i, c in enumerate(classes)}
        sy = np.array([remap[c] for c in support_y])
        se = self.embedder_(check_array(support_X))
        qe = self.embedder_(check_array(query_X))
        protos = self._prototypes(se, sy, len(classes))
        logits = self._logits(qe, protos).data
        return classes[logits.argmax(axis=1)]


class MatchingNetwork(BaseEstimator):
    """Classify a query by ATTENTION over the whole support set (Vinyals, 2016).

    Instead of collapsing each class to one prototype, a matching network keeps
    every support example and lets a query VOTE: it computes a softmax attention
    over the cosine similarity to each support embedding, then predicts the
    attention-weighted mixture of their labels. Keeping individual examples (rather
    than a mean) lets it represent multimodal classes, and the whole thing is still
    trained episodically end to end.
    """

    def __init__(self, hidden=32, embed_dim=16, n_way=3, k_shot=5, n_query=5,
                 episodes=300, lr=0.01, random_state=None):
        self.hidden = hidden; self.embed_dim = embed_dim
        self.n_way = n_way; self.k_shot = k_shot; self.n_query = n_query
        self.episodes = episodes; self.lr = lr; self.random_state = random_state

    def _attention_logits(self, qe, se, sy_onehot):
        qn = qe / ((qe * qe).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        sn = se / ((se * se).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        sims = qn @ sn.transpose()                       # cosine similarities
        attn = F.softmax(sims, axis=1)
        probs = attn @ Tensor(sy_onehot)                 # weighted label vote
        return (probs + 1e-9).log()

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        rng = check_random_state(self.random_state)
        classes = np.unique(y)
        self.embedder_ = _Embedder(X.shape[1], self.hidden, self.embed_dim, rng)
        params = self.embedder_.params()
        for _ in range(self.episodes):
            sx, sy, qx, qy, n_way = _sample_episode(
                X, y, classes, self.n_way, self.k_shot, self.n_query, rng)
            se = self.embedder_(sx); qe = self.embedder_(qx)
            onehot = np.eye(n_way)[sy]
            logp = self._attention_logits(qe, se, onehot)
            loss = _nll(logp, qy)
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        return self

    def predict(self, support_X, support_y, query_X):
        support_y = np.asarray(support_y)
        classes = np.unique(support_y)
        remap = {c: i for i, c in enumerate(classes)}
        sy = np.array([remap[c] for c in support_y])
        se = self.embedder_(check_array(support_X))
        qe = self.embedder_(check_array(query_X))
        onehot = np.eye(len(classes))[sy]
        logp = self._attention_logits(qe, se, onehot).data
        return classes[logp.argmax(axis=1)]


# --- small autograd helpers ------------------------------------------------
def _stack(rows):
    from ..autograd import Tensor
    out = rows[0].reshape(1, -1)
    for r in rows[1:]:
        out = Tensor.concatenate([out, r.reshape(1, -1)], axis=0)
    return out


def _cross_entropy(logits, targets):
    logp = F.log_softmax(logits, axis=1)
    return _nll(logp, targets)


def _nll(logp, targets):
    onehot = np.eye(logp.shape[1])[np.asarray(targets)]
    return -(logp * Tensor(onehot)).sum(axis=1).mean()
