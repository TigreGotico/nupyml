"""Embeddings v2: subword, item, count-based, hyperbolic, and universal.

These join word2vec/GloVe/BPE/WordPiece/guided-categorical. Each embeds a
different kind of thing (subwords, items, hierarchies, arbitrary entities) or by a
different principle (counts, hyperbolic geometry).
"""
import numpy as np
from collections import Counter

from ..base import BaseEstimator
from ..utils import check_random_state


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ppmi_svd(sentences, dim=50, window=2, smoothing=0.75):
    """Count-based word vectors: PPMI matrix truncated by SVD.

    Before neural word2vec there were COUNT vectors, and Levy & Goldberg (2014)
    showed word2vec is implicitly factorising one. Build the word-context
    co-occurrence counts within a window; convert to POSITIVE POINTWISE MUTUAL
    INFORMATION ``max(log( P(w,c) / (P(w)P(c)) ), 0)`` -- which measures how much
    more often two words co-occur than chance, clamped at zero; then take the top
    ``dim`` singular vectors. The result rivals word2vec, is deterministic, and
    needs no training loop. ``smoothing`` raises context counts to a power (the
    context-distribution smoothing that improves rare-word vectors).

    Returns (vocab list, vectors dict word -> vector).
    """
    counts = Counter()
    ctx_counts = Counter()
    word_counts = Counter()
    total = 0
    for sent in sentences:
        for i, w in enumerate(sent):
            word_counts[w] += 1
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if j != i:
                    counts[(w, sent[j])] += 1
                    ctx_counts[sent[j]] += 1
                    total += 1
    vocab = sorted(word_counts)
    idx = {w: k for k, w in enumerate(vocab)}
    V = len(vocab)
    ctx_sm = {c: ctx_counts[c] ** smoothing for c in vocab}
    Z = sum(ctx_sm.values())
    M = np.zeros((V, V))
    for (w, c), n in counts.items():
        p_wc = n / total
        p_w = word_counts[w] / total
        p_c = ctx_sm[c] / Z
        M[idx[w], idx[c]] = max(np.log(p_wc / (p_w * p_c) + 1e-12), 0.0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(s[:dim])
    return vocab, {w: emb[idx[w]] for w in vocab}


class Item2Vec(BaseEstimator):
    """word2vec's skip-gram applied to ITEM sets (Barkan & Koenigstein, 2016).

    Recommendation data is baskets/sessions -- SETS of co-purchased or co-played
    items with no natural order. Item2vec drops word2vec's window and treats every
    pair of items in a basket as a (target, context) pair, training skip-gram with
    negative sampling. Items that co-occur end up with similar vectors, so nearest
    neighbours in the embedding are "bought together" recommendations -- learned
    from co-occurrence alone, no ratings. ``most_similar`` returns the nearest
    items.
    """

    def __init__(self, dim=32, epochs=20, lr=0.025, n_negative=5, random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def fit(self, baskets):
        rng = check_random_state(self.random_state)
        items = sorted({i for b in baskets for i in b})
        self.idx_ = {it: k for k, it in enumerate(items)}
        self.items_ = items
        n = len(items)
        self.W_ = rng.normal(0, 0.1, (n, self.dim))
        self.C_ = rng.normal(0, 0.1, (n, self.dim))
        freq = np.array([sum(b.count(it) for b in baskets) for it in items], float)
        neg_p = freq ** 0.75; neg_p /= neg_p.sum()
        pairs = [(self.idx_[a], self.idx_[c]) for b in baskets
                 for a in b for c in b if a != c]
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for t, c in pairs:
                negs = rng.choice(n, self.n_negative, p=neg_p)
                self._sgns_step(t, c, negs)
        self.embedding_ = self.W_
        return self

    def _sgns_step(self, t, c, negs):
        lr = self.lr
        # positive pair
        g = (_sigmoid(self.W_[t] @ self.C_[c]) - 1.0)
        grad_t = g * self.C_[c]
        self.C_[c] -= lr * g * self.W_[t]
        for k in negs:                                 # negative samples
            gk = _sigmoid(self.W_[t] @ self.C_[k])
            grad_t += gk * self.C_[k]
            self.C_[k] -= lr * gk * self.W_[t]
        self.W_[t] -= lr * grad_t

    def most_similar(self, item, k=5):
        v = self.W_[self.idx_[item]]
        sims = self.W_ @ v / (np.linalg.norm(self.W_, axis=1) * np.linalg.norm(v) + 1e-9)
        order = np.argsort(-sims)
        return [self.items_[j] for j in order if self.items_[j] != item][:k]


class FastText(BaseEstimator):
    """Word vectors from character n-grams -- OOV-robust (Bojanowski et al., 2017).

    word2vec has one vector per word and is helpless on a word it never saw.
    FastText represents each word as the sum of its CHARACTER n-gram vectors (plus
    a whole-word vector), and trains those subword vectors with skip-gram. So an
    unseen or misspelled word still gets a sensible vector from its subwords, and
    morphologically related words ("run", "running", "runner") share n-grams and
    end up close -- a large win for rare words and rich morphology. Subwords are
    hashed into ``bucket`` buckets to bound memory.
    """

    def __init__(self, dim=32, min_n=3, max_n=5, bucket=2000, epochs=20, lr=0.025,
                 n_negative=5, window=2, random_state=None):
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.window = window
        self.random_state = random_state

    def _subwords(self, word):
        w = "<" + word + ">"
        grams = [w]
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(w) - n + 1):
                grams.append(w[i:i + n])
        return [hash(g) % self.bucket for g in grams]

    def fit(self, sentences):
        rng = check_random_state(self.random_state)
        vocab = sorted({w for s in sentences for w in s})
        self.vidx_ = {w: k for k, w in enumerate(vocab)}
        self.vocab_ = vocab
        self.sub_ = rng.normal(0, 0.1, (self.bucket, self.dim))   # subword vectors
        self.ctx_ = rng.normal(0, 0.1, (len(vocab), self.dim))    # context vectors
        freq = np.ones(len(vocab))
        neg_p = freq / freq.sum()
        for _ in range(self.epochs):
            for sent in sentences:
                for i, w in enumerate(sent):
                    v = self._word_vec(w)
                    lo, hi = max(0, i - self.window), min(len(sent), i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        c = self.vidx_[sent[j]]
                        negs = rng.choice(len(vocab), self.n_negative, p=neg_p)
                        self._step(w, v, c, negs)
        return self

    def _word_vec(self, word):
        return self.sub_[self._subwords(word)].sum(axis=0)

    def _step(self, word, v, c, negs):
        subs = self._subwords(word)
        grad = (_sigmoid(v @ self.ctx_[c]) - 1.0) * self.ctx_[c]
        self.ctx_[c] -= self.lr * (_sigmoid(v @ self.ctx_[c]) - 1.0) * v
        for k in negs:
            gk = _sigmoid(v @ self.ctx_[k])
            grad += gk * self.ctx_[k]
            self.ctx_[k] -= self.lr * gk * v
        for s in subs:                                 # spread gradient over subwords
            self.sub_[s] -= self.lr * grad / len(subs)

    def get_vector(self, word):
        """Works for ANY word (in-vocab or not) via its character n-grams."""
        return self._word_vec(word)


class PoincareEmbedding(BaseEstimator):
    """Embed a HIERARCHY in HYPERBOLIC space (Nickel & Kiela, 2017).

    Trees do not fit in Euclidean space: the number of nodes grows exponentially
    with depth, but Euclidean volume grows only polynomially, so a tree gets
    badly distorted. Hyperbolic space's volume grows EXPONENTIALLY, matching the
    tree, so a hierarchy embeds with tiny distortion in just 2 dimensions. Points
    live in the Poincare BALL; distance uses the hyperbolic metric, and general
    concepts naturally settle near the ORIGIN while specific ones push toward the
    boundary. Trained by Riemannian SGD on related/unrelated pairs.

    ``fit`` takes hierarchy edges ``(parent, child)``; ``distance`` gives the
    hyperbolic distance.
    """

    def __init__(self, dim=2, epochs=100, lr=0.1, n_negative=5, random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.n_negative = n_negative
        self.random_state = random_state

    def _dist(self, u, v):
        sq = np.sum((u - v) ** 2)
        nu = np.sum(u ** 2); nv = np.sum(v ** 2)
        return np.arccosh(1 + 2 * sq / ((1 - nu) * (1 - nv) + 1e-9))

    def _dist_grad(self, u, v):
        """Euclidean gradient of the Poincare distance d(u,v) w.r.t. u."""
        a = 1 - np.sum(u ** 2)
        b = 1 - np.sum(v ** 2)
        uv = np.sum((u - v) ** 2)
        gamma = 1 + 2 * uv / (a * b + 1e-9)
        c = 4 / (b * np.sqrt(gamma ** 2 - 1) + 1e-9)
        return c * ((np.sum(v ** 2) - 2 * (u @ v) + 1) / (a ** 2 + 1e-9) * u - v / (a + 1e-9))

    def fit(self, edges):
        rng = check_random_state(self.random_state)
        nodes = sorted({n for e in edges for n in e})
        self.idx_ = {n: k for k, n in enumerate(nodes)}
        self.nodes_ = nodes
        N = len(nodes)
        self.emb_ = rng.uniform(-0.05, 0.05, (N, self.dim))
        E = [(self.idx_[a], self.idx_[b]) for a, b in edges]
        for _ in range(self.epochs):
            rng.shuffle(E)
            for u, v in E:
                negs = [rng.randint(N) for _ in range(self.n_negative)]
                self._step(u, v, negs)
        self.embedding_ = self.emb_
        return self

    def _riemannian(self, node, egrad):
        scale = ((1 - np.sum(self.emb_[node] ** 2)) ** 2) / 4   # inverse metric
        self.emb_[node] = self._project(self.emb_[node] - self.lr * scale * egrad)

    def _step(self, u, v, negs):
        # softmax over -distance to {positive v} + negatives; pull v in, push negs out
        cands = [v] + [n for n in negs if n != u]
        d = np.array([self._dist(self.emb_[u], self.emb_[c]) for c in cands])
        expd = np.exp(-d); p = expd / expd.sum()
        grad_u = np.zeros(self.dim)
        for rank, c in enumerate(cands):
            coeff = -(p[rank] - (1.0 if rank == 0 else 0.0))   # dLoss/d(-d) chain
            gu = self._dist_grad(self.emb_[u], self.emb_[c])
            gc = self._dist_grad(self.emb_[c], self.emb_[u])
            grad_u += coeff * gu
            self._riemannian(c, coeff * gc)            # move the endpoint too
        self._riemannian(u, grad_u)

    def _project(self, x):
        norm = np.linalg.norm(x)
        return x / norm * (1 - 1e-5) if norm >= 1 else x

    def distance(self, a, b):
        return float(self._dist(self.emb_[self.idx_[a]], self.emb_[self.idx_[b]]))

    def norm(self, a):
        return float(np.linalg.norm(self.emb_[self.idx_[a]]))


class StarSpace(BaseEstimator):
    """Embed EVERYTHING into one space and compare by similarity (Wu et al., 2018).

    A single model for many tasks: represent both an entity (a bag of features --
    words, tags) and a label as vectors in the SAME space, and train so a
    positive (entity, label) pair scores higher than a sampled negative by a margin
    (a ranking loss). Classification becomes "nearest label in the shared space";
    the same recipe does retrieval, recommendation, or entity similarity just by
    changing what the pairs are. ``fit(bags, labels)`` where each bag is a list of
    feature ids; ``predict`` returns the nearest label.
    """

    def __init__(self, dim=32, epochs=20, lr=0.05, margin=0.1, n_negative=5,
                 random_state=None):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.n_negative = n_negative
        self.random_state = random_state

    def _bag_vec(self, bag):
        if not bag:
            return np.zeros(self.dim)
        return self.feat_[list(bag)].mean(axis=0)

    def fit(self, bags, labels):
        rng = check_random_state(self.random_state)
        n_feat = max(f for b in bags for f in b) + 1
        self.classes_ = sorted(set(labels))
        self.lidx_ = {c: k for k, c in enumerate(self.classes_)}
        self.feat_ = rng.normal(0, 0.1, (n_feat, self.dim))
        self.label_ = rng.normal(0, 0.1, (len(self.classes_), self.dim))
        y = [self.lidx_[l] for l in labels]
        for _ in range(self.epochs):
            order = rng.permutation(len(bags))
            for i in order:
                v = self._bag_vec(bags[i])
                pos = y[i]
                for _ in range(self.n_negative):
                    neg = rng.randint(len(self.classes_))
                    if neg == pos:
                        continue
                    s_pos = v @ self.label_[pos]
                    s_neg = v @ self.label_[neg]
                    if s_pos - s_neg < self.margin:    # margin ranking violation
                        self.label_[pos] += self.lr * v
                        self.label_[neg] -= self.lr * v
                        for f in bags[i]:
                            self.feat_[f] += self.lr * (self.label_[pos] - self.label_[neg]) / len(bags[i])
        return self

    def predict(self, bags):
        out = []
        for b in bags:
            v = self._bag_vec(b)
            out.append(self.classes_[int(np.argmax(self.label_ @ v))])
        return np.array(out)


__all__ = ["ppmi_svd", "Item2Vec", "FastText", "PoincareEmbedding", "StarSpace"]
