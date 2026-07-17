"""Topic models and text ranking: LDA, LSA, BM25, Doc2Vec.

The corpus is built from two obviously-distinct themes (animals vs computers) so
each model can be held to recovering that structure -- separated topics, a
relevance ranking that respects exact terms, document vectors that cluster by
theme.
"""
import numpy as np
import pytest

from nupyml.topic import (LatentDirichletAllocation, topic_coherence,
                          LatentSemanticAnalysis, BM25, Doc2Vec)

ANIMAL_WORDS = {"cat", "dog", "animal", "pet", "fur", "paw", "tail"}
TECH_WORDS = {"computer", "code", "software", "data", "byte", "chip"}


@pytest.fixture
def two_topic_corpus():
    animals = ["cat dog animal pet fur paw", "dog cat pet animal tail",
               "animal fur cat dog paw pet"]
    tech = ["computer code software data byte", "software data code computer chip",
            "data byte software code computer"]
    strings = (animals + tech) * 8
    return strings, [s.split() for s in strings]


# --- LDA ------------------------------------------------------------------

def test_lda_recovers_separated_topics(two_topic_corpus):
    """Two distinct themes should come out as two pure topics."""
    _, docs = two_topic_corpus
    lda = LatentDirichletAllocation(n_topics=2, n_iter=200, random_state=0).fit(docs)
    t0, t1 = set(lda.top_words(0, 4)), set(lda.top_words(1, 4))
    pure = ((len(t0 & ANIMAL_WORDS) >= 3 and len(t1 & TECH_WORDS) >= 3)
            or (len(t0 & TECH_WORDS) >= 3 and len(t1 & ANIMAL_WORDS) >= 3))
    assert pure


def test_lda_assigns_documents_to_one_topic(two_topic_corpus):
    """With a sparse alpha, an animal document should be almost entirely one
    topic -- the sparsity the default alpha=0.1 buys."""
    _, docs = two_topic_corpus
    lda = LatentDirichletAllocation(n_topics=2, n_iter=200, random_state=0).fit(docs)
    mix = lda.transform(docs[:6])
    # each of the first six docs (3 animal, 3 tech) is dominated by one topic
    assert np.all(mix.max(axis=1) > 0.75)


def test_lda_large_alpha_blurs_the_mixtures(two_topic_corpus):
    """The alpha lesson: a large document-topic prior forces uniform mixtures on
    short documents, which is exactly why the default is small."""
    _, docs = two_topic_corpus
    sparse = LatentDirichletAllocation(n_topics=2, alpha=0.1, n_iter=150,
                                       random_state=0).fit(docs)
    blurred = LatentDirichletAllocation(n_topics=2, alpha=50.0, n_iter=150,
                                        random_state=0).fit(docs)
    assert sparse.transform(docs[:1]).max() > blurred.transform(docs[:1]).max()


def test_lda_topic_word_rows_are_distributions(two_topic_corpus):
    _, docs = two_topic_corpus
    lda = LatentDirichletAllocation(n_topics=2, n_iter=100, random_state=0).fit(docs)
    assert np.allclose(lda.topic_word_.sum(axis=1), 1.0)


def test_topic_coherence_prefers_a_real_topic(two_topic_corpus):
    """A coherent topic (words that co-occur) scores higher than a jumbled one."""
    _, docs = two_topic_corpus
    good = ["cat", "dog", "animal", "pet"]           # all co-occur
    bad = ["cat", "byte", "paw", "software"]         # never together
    assert topic_coherence([good], docs) > topic_coherence([bad], docs)


# --- LSA ------------------------------------------------------------------

def test_lsa_topic_is_thematically_coherent(two_topic_corpus):
    strings, _ = two_topic_corpus
    lsa = LatentSemanticAnalysis(n_topics=2).fit(strings)
    words = set(lsa.top_words(0, 5))
    # the leading topic axis should be dominated by ONE theme
    assert len(words & ANIMAL_WORDS) >= 4 or len(words & TECH_WORDS) >= 4


def test_lsa_projects_into_topic_space(two_topic_corpus):
    strings, _ = two_topic_corpus
    lsa = LatentSemanticAnalysis(n_topics=2).fit(strings)
    proj = lsa.transform(strings)
    assert proj.shape == (len(strings), 2)


def test_lsa_separates_the_two_themes(two_topic_corpus):
    """Animal and tech documents should occupy different regions of topic space."""
    strings, _ = two_topic_corpus
    lsa = LatentSemanticAnalysis(n_topics=2).fit(strings)
    proj = lsa.transform(strings)
    # docs 0,1,2 (animal) vs 3,4,5 (tech) in the first repetition
    animal_centroid = proj[[0, 1, 2]].mean(axis=0)
    tech_centroid = proj[[3, 4, 5]].mean(axis=0)
    assert np.linalg.norm(animal_centroid - tech_centroid) > 0.1


# --- BM25 -----------------------------------------------------------------

@pytest.fixture
def search_corpus():
    return [d.split() for d in [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cat cat cat kitten feline",       # very much about cats
        "software runs on computers"]]


def test_bm25_ranks_the_most_relevant_document(search_corpus):
    bm = BM25().fit(search_corpus)
    ranking = bm.rank(["cat"])
    assert ranking[0] == 2                 # the cat-heavy document wins


def test_bm25_saturates_term_frequency(search_corpus):
    """The k1 knob: three 'cat's are worth well less than three times one 'cat'
    -- the saturation that stops keyword stuffing."""
    bm = BM25(k1=1.5).fit(search_corpus)
    scores = bm.score(["cat"])
    # doc 2 has 3 cats, doc 0 has 1; its score is higher but far from 3x
    assert scores[2] > scores[0]
    assert scores[2] < 3 * scores[0]


def test_bm25_length_normalization_penalizes_padding():
    """Two documents with one 'cat' each, one padded with filler: the shorter,
    denser one should score higher for 'cat' (the b knob at work)."""
    corpus = [["cat"], ["cat"] + ["filler"] * 50]
    bm = BM25(b=0.75).fit(corpus)
    scores = bm.score(["cat"])
    assert scores[0] > scores[1]


def test_bm25_ignores_absent_terms(search_corpus):
    bm = BM25().fit(search_corpus)
    assert np.all(bm.score(["nonexistentword"]) == 0)


# --- Doc2Vec --------------------------------------------------------------

def test_doc2vec_clusters_documents_by_theme(two_topic_corpus):
    """Documents about the same theme should have more similar vectors than
    documents about different themes."""
    _, docs = two_topic_corpus
    d2v = Doc2Vec(n_dim=30, n_epochs=40, random_state=0).fit(docs)
    # docs 0,1,2 are animal; 3,4,5 are tech (first repetition)
    same_theme = d2v.similarity(0, 1)
    cross_theme = d2v.similarity(0, 3)
    assert same_theme > cross_theme


def test_doc2vec_vectors_have_the_requested_dimension(two_topic_corpus):
    _, docs = two_topic_corpus
    d2v = Doc2Vec(n_dim=24, n_epochs=10, random_state=0).fit(docs)
    assert d2v.document_vectors_.shape == (len(docs), 24)


def test_doc2vec_most_similar_returns_same_theme(two_topic_corpus):
    _, docs = two_topic_corpus
    d2v = Doc2Vec(n_dim=30, n_epochs=40, random_state=0).fit(docs)
    # the nearest neighbours of an animal doc (index 0) should be animal docs.
    # animal docs are indices 0,1,2,6,7,8,... (every block of 6 starts with 3)
    neighbours = [j for j, _ in d2v.most_similar(0, topn=5)]
    animal_idx = {i for i in range(len(docs)) if i % 6 < 3}
    assert sum(j in animal_idx for j in neighbours) >= 4
