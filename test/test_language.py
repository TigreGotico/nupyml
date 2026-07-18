"""G10: n-gram language model, TextRank, SIF embeddings, GRU4Rec.

The LM must give low perplexity to in-distribution text and high to out-of-
distribution (and never infinite, thanks to smoothing); TextRank must surface the
most-connected word/sentence; SIF must remove the common component; GRU4Rec must
learn the next item in patterned sessions.
"""
import numpy as np
import pytest

from nupyml.topic import (NGramLanguageModel, textrank_keywords,
                         textrank_summary, SIFEmbedding)
from nupyml.recommend import GRU4Rec


@pytest.fixture
def corpus():
    return [["the", "cat", "sat", "on", "the", "mat"]] * 20


def test_kneser_ney_scores_in_distribution_lower(corpus):
    lm = NGramLanguageModel(n=2, smoothing="kneser_ney").fit(corpus)
    ppl_in = lm.perplexity(corpus)
    ppl_out = lm.perplexity([["mat", "the", "cat", "zzz"]])
    assert ppl_in < ppl_out
    assert np.isfinite(ppl_out)                       # smoothing -> no infinities


def test_add_k_and_kneser_ney_both_finite(corpus):
    for sm in ("add_k", "kneser_ney"):
        lm = NGramLanguageModel(n=2, smoothing=sm, k=1.0).fit(corpus)
        # an entirely unseen word still gets finite (non-zero) probability
        p = lm.prob("unseen_word", ["the"])
        assert p > 0 and np.isfinite(p)


def test_ngram_generate_produces_known_tokens(corpus):
    lm = NGramLanguageModel(n=2).fit(corpus)
    gen = lm.generate(max_len=10, random_state=0)
    assert all(tok in lm.vocab_ for tok in gen)


def test_textrank_keywords_finds_the_central_word():
    tokens = ("data science machine learning data model learning "
              "algorithm data model science").split()
    kws = textrank_keywords(tokens, top_k=3)
    assert "data" in kws                              # the most-connected word


def test_textrank_summary_returns_ordered_subset():
    sents = [["cats", "are", "great", "pets"],
             ["dogs", "are", "loyal", "pets"],
             ["the", "stock", "market", "fell", "sharply"],
             ["cats", "and", "dogs", "are", "pets"]]
    summary = textrank_summary(sents, top_k=2)
    assert len(summary) == 2
    assert all(s in sents for s in summary)
    # the off-topic finance sentence should NOT be the most central
    assert ["the", "stock", "market", "fell", "sharply"] not in summary


def test_sif_removes_the_common_component():
    rng = np.random.RandomState(0)
    sents = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"], ["c", "e", "a"]]
    vocab = set(w for s in sents for w in s)
    wv = {w: rng.randn(6) for w in vocab}
    sif = SIFEmbedding().fit(sents, wv)
    E = sif.transform(sents, wv)
    assert E.shape == (4, 6)
    # after projecting out the common direction, embeddings are ~orthogonal to it
    assert np.allclose(E @ sif.common_, 0, atol=1e-8)


def test_gru4rec_learns_session_pattern():
    # two deterministic session patterns; the model must predict the continuation
    sessions = [[0, 1, 2, 3]] * 30 + [[3, 2, 1, 0]] * 30
    g = GRU4Rec(n_items=4, embed_dim=8, hidden=16, epochs=40, random_state=0)
    g.fit(sessions)
    assert g.predict_next([0, 1]) == 2                # 0->1->2...
    assert g.predict_next([3, 2]) == 1                # 3->2->1...
    recs = g.recommend([0, 1], k=2)
    assert 2 in recs and 0 not in recs                # excludes seen items
