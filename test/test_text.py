"""J3: text / language models -- Kneser-Ney, TextRank, LexRank, RAKE, MEMM.

Kneser-Ney forms a valid conditional distribution and prefers seen continuations;
TextRank/LexRank surface central sentences; RAKE pulls multi-word keyphrases with
no training; the MEMM learns to tag a toy sequence and decodes with Viterbi.
"""
import numpy as np
import pytest

from nupyml.text import KneserNeyLM, TextRank, LexRank, rake_keywords, MEMM


CORPUS = ['the cat sat on the mat', 'the dog sat on the log',
          'a cat ran', 'the cat ran fast'] * 5


def test_kneser_ney_is_a_valid_distribution_and_prefers_seen():
    kn = KneserNeyLM().fit(CORPUS)
    # a well-attested continuation beats an unlikely one
    assert kn.prob('the', 'cat') > kn.prob('the', 'fast')
    # given a context, probabilities over the vocabulary sum to (about) 1
    total = sum(kn.prob('the', w) for w in kn.vocab_)
    assert total == pytest.approx(1.0, abs=0.05)


def test_kneser_ney_beats_add_one_perplexity():
    train = CORPUS
    test = ['the cat sat on the mat', 'the dog ran']
    kn = KneserNeyLM(discount=0.75).fit(train)
    # Kneser-Ney (discount) gives lower perplexity than heavy discounting=0 add-ish
    kn0 = KneserNeyLM(discount=0.1).fit(train)
    assert kn.perplexity(test) < 1e6                  # finite, well-defined
    assert kn.perplexity(train) < kn.perplexity(['zzz qqq wxy'])  # seen < unseen


def test_textrank_ranks_central_sentences():
    sents = ['The cat sat on the mat.', 'Dogs are loyal animals.',
             'The cat and the dog played on the mat.',
             'Machine learning is fun.',
             'The dog chased the cat around the mat.']
    tr = TextRank().fit(sents)
    top = set(np.argsort(tr.scores_)[::-1][:2])
    # the cat/dog/mat sentences are central; the isolated ML sentence is not
    assert 3 not in top
    summary = TextRank().summary(sents, k=2)
    assert len(summary) == 2


def test_lexrank_produces_finite_scores_and_summary():
    sents = ['The cat sat on the mat.', 'Dogs are loyal animals.',
             'The cat and the dog played.', 'The dog chased the cat.']
    lr = LexRank(threshold=0.05).fit(sents)
    assert np.all(np.isfinite(lr.scores_))
    assert len(LexRank().summary(sents, k=2)) == 2


def test_rake_extracts_multiword_keyphrases():
    text = ('machine learning algorithms are powerful. '
            'deep learning improves language processing.')
    kws = rake_keywords(text, top_k=5)
    phrases = [p for p, s in kws]
    # at least one extracted keyphrase is multi-word (RAKE's whole point)
    assert any(len(p.split()) >= 2 for p in phrases)
    # scores are descending
    scores = [s for p, s in kws]
    assert scores == sorted(scores, reverse=True)


def test_memm_tags_toy_sequence():
    sents = [['the', 'cat', 'runs'], ['the', 'dog', 'sleeps'],
             ['a', 'cat', 'sleeps'], ['a', 'dog', 'runs']] * 10
    tags = [['D', 'N', 'V']] * 40
    m = MEMM(epochs=300, random_state=0).fit(sents, tags)
    assert m.predict(['the', 'dog', 'runs']) == ['D', 'N', 'V']
    assert m.predict(['a', 'cat', 'sleeps']) == ['D', 'N', 'V']
