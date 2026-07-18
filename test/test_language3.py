"""L3: language v4 -- TextTiling, PMI collocations, spell corrector, chrF/METEOR,
HMM tagger.

TextTiling finds the boundary between two topics; PMI surfaces genuine collocations;
the Norvig corrector fixes typos; chrF and METEOR score identical text at ~1 and
partial overlap lower; the HMM tagger labels a toy sequence.
"""
import numpy as np
import pytest

from nupyml.text import (text_tiling, pmi_collocations, SpellCorrector, chrf,
                         meteor, HMMTagger)


def test_text_tiling_finds_topic_boundary():
    topic_a = ' '.join(['cat dog pet animal fur paws tail'] * 8)
    topic_b = ' '.join(['stock market price trade money invest bond'] * 8)
    doc = topic_a + ' ' + topic_b
    join = len(topic_a.split())
    bounds = text_tiling(doc, block_size=8, gap_step=4)
    assert any(abs(b - join) < 12 for b in bounds)       # a boundary near the join


def test_pmi_surfaces_collocations():
    corpus = ['machine learning is great', 'i love machine learning',
              'deep learning and machine learning', 'new york city is big',
              'new york is great'] * 5
    pmi = pmi_collocations(corpus, min_count=3, top_k=6)
    pairs = {' '.join(p) for p, s in pmi}
    assert 'new york' in pairs                            # a true collocation ranks high


def test_spell_corrector_fixes_typos():
    sc = SpellCorrector().fit(['the quick brown fox spelling correction is useful']
                              * 20 + ['spelling is important'] * 10)
    assert sc.correct('speling') == 'spelling'
    assert sc.correct('korrection') == 'correction'
    assert sc.correct('the') == 'the'                    # a correct word is unchanged


def test_chrf_scores():
    assert chrf('the cat sat', 'the cat sat') == pytest.approx(1.0)
    assert 0 < chrf('color', 'colour') < 1.0             # shares character n-grams
    assert chrf('cat', 'xyz') == 0.0


def test_meteor_scores():
    assert meteor('the cat sat on the mat', 'the cat sat on the mat') > 0.95
    partial = meteor('the cat sat on the mat', 'cat the mat sat')
    assert 0 < partial < 0.95


def test_hmm_tagger():
    sents = [['the', 'cat', 'runs'], ['the', 'dog', 'sleeps'],
             ['a', 'cat', 'sleeps'], ['a', 'dog', 'runs']] * 10
    tags = [['D', 'N', 'V']] * 40
    hmm = HMMTagger().fit(sents, tags)
    assert hmm.predict(['the', 'dog', 'runs']) == ['D', 'N', 'V']
    assert hmm.predict(['a', 'cat', 'sleeps']) == ['D', 'N', 'V']
