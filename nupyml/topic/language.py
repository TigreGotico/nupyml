"""Language models and graph/embedding NLP: n-gram LM, TextRank, SIF.

These join the topic models (LDA/LSA) and ranking (BM25). A language model scores
and generates sequences; TextRank extracts keywords/summaries without training;
SIF turns word vectors into a strong sentence embedding.
"""
import numpy as np
from collections import defaultdict, Counter


class NGramLanguageModel:
    """An n-gram language model with add-k or Kneser-Ney smoothing.

    THE MODEL, AND WHY SMOOTHING IS EVERYTHING
    ------------------------------------------
    Estimate ``P(word | previous n-1 words)`` from counts. The catch: most n-grams
    are NEVER seen in training, and assigning them probability zero makes any test
    text containing one infinitely surprising (zero probability, infinite
    perplexity). Smoothing redistributes mass to the unseen:

    * **add-k** -- add a pseudo-count ``k`` to every n-gram. Simple, but crude: it
      steals too much from frequent events and treats all unseen n-grams alike.
    * **Kneser-Ney** -- the best classical smoothing. Its insight: an unseen word's
      backoff probability should reflect how many DIFFERENT contexts it appears in,
      not its raw frequency. "Francisco" is frequent but almost only after "San",
      so it is a poor guess in a new context; Kneser-Ney's CONTINUATION
      probability captures exactly that. Implemented here for the bigram model.

    ``fit`` takes token sequences (lists of tokens); ``perplexity`` scores held-out
    text; ``generate`` samples.
    """

    def __init__(self, n=2, smoothing="kneser_ney", k=1.0, discount=0.75):
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.discount = discount

    def fit(self, sequences):
        self.vocab_ = set()
        self.ngrams_ = defaultdict(Counter)          # context -> Counter(next word)
        self.context_counts_ = Counter()
        for seq in sequences:
            seq = ["<s>"] * (self.n - 1) + list(seq) + ["</s>"]
            self.vocab_.update(seq)
            for i in range(self.n - 1, len(seq)):
                ctx = tuple(seq[i - self.n + 1:i])
                self.ngrams_[ctx][seq[i]] += 1
                self.context_counts_[ctx] += 1
        self.V_ = len(self.vocab_)
        if self.smoothing == "kneser_ney" and self.n == 2:
            # continuation counts: in how many distinct contexts does w appear?
            self._cont = Counter()
            self._total_bigrams = 0
            for ctx, counter in self.ngrams_.items():
                for w in counter:
                    self._cont[w] += 1
                    self._total_bigrams += 1
        return self

    def prob(self, word, context):
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        counter = self.ngrams_.get(context, Counter())
        ctx_total = self.context_counts_.get(context, 0)
        if self.smoothing == "kneser_ney" and self.n == 2:
            d = self.discount
            if word in self.vocab_:
                cont = self._cont.get(word, 0) / max(1, self._total_bigrams)
            else:
                cont = 1.0 / self.V_                  # OOV floor: back off to uniform
            if ctx_total == 0:
                return max(cont, 1e-10)               # unseen context -> continuation
            lam = d * len(counter) / ctx_total        # normalised backoff weight
            return max(counter[word] - d, 0) / ctx_total + lam * cont
        # add-k
        return (counter[word] + self.k) / (ctx_total + self.k * self.V_)

    def perplexity(self, sequences):
        log_sum, n_tokens = 0.0, 0
        for seq in sequences:
            seq2 = ["<s>"] * (self.n - 1) + list(seq) + ["</s>"]
            for i in range(self.n - 1, len(seq2)):
                p = self.prob(seq2[i], seq2[i - self.n + 1:i])
                log_sum += np.log(max(p, 1e-12))
                n_tokens += 1
        return float(np.exp(-log_sum / max(1, n_tokens)))

    def generate(self, max_len=20, random_state=None):
        from ..utils import check_random_state
        rng = check_random_state(random_state)
        context = ["<s>"] * (self.n - 1)
        out = []
        vocab = sorted(self.vocab_)
        for _ in range(max_len):
            probs = np.array([self.prob(w, context) for w in vocab])
            probs /= probs.sum()
            w = vocab[rng.choice(len(vocab), p=probs)]
            if w == "</s>":
                break
            if w != "<s>":
                out.append(w)
            context = (context + [w])[-(self.n - 1):] if self.n > 1 else []
        return out


def textrank_keywords(tokens, window=4, top_k=5, damping=0.85):
    """Extract keywords by running PageRank on a word CO-OCCURRENCE graph.

    THE IDEA (Mihalcea & Tarau, 2004)
    ---------------------------------
    Build a graph whose nodes are words and whose edges link words that co-occur
    within a sliding window; a word is IMPORTANT if it co-occurs with other
    important words -- exactly PageRank's recursive definition, applied to text.
    No training, no labels: importance emerges from the co-occurrence structure
    alone. Returns the ``top_k`` words by PageRank score.
    """
    from ..graph import pagerank
    vocab = sorted(set(tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    A = np.zeros((len(vocab), len(vocab)))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            a, b = idx[tokens[i]], idx[tokens[j]]
            if a != b:
                A[a, b] += 1; A[b, a] += 1
    scores = pagerank(A, damping=damping)
    order = np.argsort(scores)[::-1][:top_k]
    return [vocab[i] for i in order]


def textrank_summary(sentences, top_k=3, damping=0.85):
    """Extractive summarisation: PageRank over a sentence-SIMILARITY graph.

    Same principle as keyword TextRank, one level up: nodes are sentences, edge
    weights are their word-overlap similarity, and the highest-PageRank sentences
    are the most CENTRAL -- the ones most representative of the whole document.
    Returns the ``top_k`` sentences in their original order. ``sentences`` is a
    list of token lists.
    """
    from ..graph import pagerank
    n = len(sentences)
    sets = [set(s) for s in sentences]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            denom = np.log(len(sets[i]) + 1) + np.log(len(sets[j]) + 1)
            sim = inter / denom if denom > 0 else 0.0
            A[i, j] = A[j, i] = sim
    scores = pagerank(A, damping=damping)
    chosen = sorted(np.argsort(scores)[::-1][:top_k])
    return [sentences[i] for i in chosen]


class SIFEmbedding:
    """Smooth Inverse Frequency sentence embeddings (Arora et al., 2017).

    THE "SIMPLE BASELINE THAT IS HARD TO BEAT"
    ------------------------------------------
    Averaging word vectors makes a decent sentence vector, but frequent words
    ("the", "is") dominate the average and add no meaning. SIF fixes this in two
    steps: (1) weight each word by ``a / (a + p(w))`` so common words are
    down-weighted (smooth inverse frequency), then (2) remove the projection onto
    the first PRINCIPAL COMPONENT of all sentence vectors -- which turns out to
    encode exactly the common "syntax/frequency" direction, not meaning. The
    result rivals far heavier sentence encoders. ``fit`` learns word frequencies
    and the common component; ``transform`` embeds sentences given word vectors.
    """

    def __init__(self, a=1e-3):
        self.a = a

    def fit(self, sentences, word_vectors):
        counts = Counter(w for s in sentences for w in s)
        total = sum(counts.values())
        self.freq_ = {w: c / total for w, c in counts.items()}
        self.dim_ = len(next(iter(word_vectors.values())))
        emb = self._weighted(sentences, word_vectors)
        # the first principal component of the sentence vectors ~ the common
        # discourse direction; store it to project out
        u, s, vt = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
        self.common_ = vt[0]
        return self

    def _weighted(self, sentences, word_vectors):
        out = np.zeros((len(sentences), self.dim_))
        for i, s in enumerate(sentences):
            vecs = [word_vectors[w] * (self.a / (self.a + self.freq_.get(w, 0)))
                    for w in s if w in word_vectors]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out

    def transform(self, sentences, word_vectors):
        emb = self._weighted(sentences, word_vectors)
        # remove the common component: emb - (emb . u) u
        return emb - np.outer(emb @ self.common_, self.common_)


__all__ = ["NGramLanguageModel", "textrank_keywords", "textrank_summary",
           "SIFEmbedding"]
