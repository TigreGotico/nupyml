"""Latent Dirichlet Allocation by collapsed Gibbs sampling.

THE GENERATIVE STORY
--------------------
LDA imagines each document was written like this:

1. pick a mixture of topics for the document (from a Dirichlet prior);
2. for each word position, pick a topic from that mixture, then pick a word from
   that topic's word-distribution.

Fitting REVERSES the story: given only the words, recover the topic each word was
drawn from, and hence the topic-word and document-topic distributions. The
Dirichlet priors (``alpha`` on document-topic, ``beta`` on topic-word) are what
make the mixtures SPARSE -- a document is about a few topics, a topic emphasises a
few words -- which is what makes the result interpretable rather than mush.

WHY COLLAPSED GIBBS
-------------------
The document-topic and topic-word distributions can be integrated out
analytically (they are Dirichlet-multinomial conjugate), leaving only the
per-word topic ASSIGNMENTS to sample. So the sampler holds just one integer per
word -- its current topic -- and repeatedly redraws it from::

    P(topic = k | everything else) proportional to
        (how much this DOC already uses k) x (how much k likes this WORD)

Both factors are simple counts, so a sweep is fast, and the assignments converge
to samples from the posterior. The two count matrices ARE the model. This is
Griffiths & Steyvers' method, and it is both the easiest LDA to implement and the
clearest to read -- the two competing pulls above are the whole algorithm.

Blei, Ng & Jordan (2003); Griffiths & Steyvers (2004).
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_random_state


class LatentDirichletAllocation(BaseEstimator, TransformerMixin):
    """Discover topics in a corpus of tokenised documents.

    ``fit`` takes a list of documents, each a list of string tokens. After
    fitting, ``topic_word_`` is (n_topics, vocab) and ``transform`` returns each
    document's topic mixture. ``top_words(k)`` prints a topic as its heaviest
    words -- the human-readable form of a topic.
    """

    def __init__(self, n_topics=10, alpha=0.1, beta=0.01, n_iter=200,
                 random_state=None):
        # alpha is the document-topic prior, and it is a SPARSITY knob: a small
        # alpha (the default 0.1) lets each document commit to a few topics, which
        # is what gives clean, separated mixtures. The textbook 50/n_topics
        # default assumes long documents -- on short ones (a few words) it swamps
        # the word evidence and forces every document to a uniform ~1/K mixture,
        # so the topics never specialise. When in doubt, keep alpha small.
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta                # topic-word prior (also sparse by default)
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, documents, y=None):
        rng = check_random_state(self.random_state)
        K = self.n_topics
        alpha = self.alpha

        # build the vocabulary and index every word token
        self.vocabulary_ = {}
        docs = []
        for doc in documents:
            idx = []
            for w in doc:
                if w not in self.vocabulary_:
                    self.vocabulary_[w] = len(self.vocabulary_)
                idx.append(self.vocabulary_[w])
            docs.append(idx)
        V = len(self.vocabulary_)
        self.index_to_word_ = {i: w for w, i in self.vocabulary_.items()}

        # the count matrices the sampler maintains
        n_dk = np.zeros((len(docs), K))         # topic counts per document
        n_kw = np.zeros((K, V))                 # word counts per topic
        n_k = np.zeros(K)                       # total words per topic
        # assignments[d] holds the current topic of each word in document d
        assignments = []
        for d, doc in enumerate(docs):
            z = rng.randint(0, K, size=len(doc))   # random initial topics
            assignments.append(z)
            for w, k in zip(doc, z):
                n_dk[d, k] += 1
                n_kw[k, w] += 1
                n_k[k] += 1

        for _ in range(self.n_iter):
            for d, doc in enumerate(docs):
                z = assignments[d]
                for i, w in enumerate(doc):
                    k = z[i]
                    # remove this word's current assignment ("leave-one-out")
                    n_dk[d, k] -= 1; n_kw[k, w] -= 1; n_k[k] -= 1
                    # the two pulls: how much the doc uses k, times how much k
                    # likes word w -- with the Dirichlet pseudo-counts added
                    p = (n_dk[d] + alpha) * (n_kw[:, w] + self.beta) \
                        / (n_k + V * self.beta)
                    p /= p.sum()
                    k_new = rng.choice(K, p=p)
                    # put it back under its resampled topic
                    z[i] = k_new
                    n_dk[d, k_new] += 1; n_kw[k_new, w] += 1; n_k[k_new] += 1

        # the fitted distributions, smoothed by the priors and normalised
        self.topic_word_ = (n_kw + self.beta)
        self.topic_word_ /= self.topic_word_.sum(axis=1, keepdims=True)
        self._doc_topic = (n_dk + alpha)
        self._doc_topic /= self._doc_topic.sum(axis=1, keepdims=True)
        self._alpha = alpha
        return self

    def transform(self, documents):
        """Topic mixture per document, holding the learned topics fixed.

        A few EM-like sweeps: given the current document-topic estimate, assign
        each word softly to topics (proportional to doc-use x topic-likes-word),
        re-accumulate, repeat. Converges to the document's mixture over the fitted
        topics without touching them.
        """
        check_is_fitted(self, "topic_word_")
        K = self.n_topics
        out = np.zeros((len(documents), K))
        for d, doc in enumerate(documents):
            idx = [self.vocabulary_[w] for w in doc if w in self.vocabulary_]
            theta = np.full(K, 1.0 / K)
            for _ in range(30):
                counts = np.full(K, self._alpha)
                for w in idx:
                    p = theta * self.topic_word_[:, w]   # soft topic responsibility
                    counts += p / (p.sum() + 1e-12)
                theta = counts / counts.sum()
            out[d] = theta
        return out

    def top_words(self, topic, n=10):
        """The ``n`` heaviest words of a topic -- its human-readable summary."""
        check_is_fitted(self, "topic_word_")
        order = np.argsort(self.topic_word_[topic])[::-1][:n]
        return [self.index_to_word_[i] for i in order]


def topic_coherence(topics, documents, top_n=10):
    """UMass coherence: do a topic's top words actually CO-OCCUR in the corpus?

    Perplexity measures how well a topic model predicts held-out text, and it is
    notorious for DISAGREEING with human judgement -- a model can score well and
    produce topics that read like word salad. Coherence scores interpretability
    directly: for a topic's top words, sum the log of how often each pair
    co-occurs in a document versus how often the rarer one appears alone. Words
    that genuinely belong to one theme co-occur, so a coherent topic scores near
    zero (less negative); an incoherent one, whose words never appear together,
    scores very negative.

    ``topics`` is a list of word-lists (each a topic's top words); ``documents``
    is the tokenised corpus.
    """
    # document frequency of each word and of each co-occurring pair
    doc_sets = [set(doc) for doc in documents]
    scores = []
    for words in topics:
        words = words[:top_n]
        total = 0.0
        for i in range(1, len(words)):
            for j in range(i):
                wi, wj = words[i], words[j]
                d_ij = sum(1 for s in doc_sets if wi in s and wj in s)
                d_j = sum(1 for s in doc_sets if wj in s)
                # smoothed log-ratio; the +1 avoids log(0) for a pair never seen
                total += np.log((d_ij + 1) / (d_j + 1e-12))
        scores.append(total)
    return float(np.mean(scores))


__all__ = ["LatentDirichletAllocation", "topic_coherence"]
