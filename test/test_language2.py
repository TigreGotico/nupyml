"""K3: language v3 -- unigram tokenizer, beam search, WMD, averaged-perceptron
tagger, BLEU/ROUGE.

The tokenizer learns subword pieces and segments words; beam search escapes a
greedy trap; Word Mover's Distance ranks a paraphrase below an unrelated sentence;
the tagger labels a toy sequence; BLEU/ROUGE are 1.0 for an identical match and
lower for partial overlap.
"""
import numpy as np
import pytest

from nupyml.text import (UnigramTokenizer, beam_search, word_movers_distance,
                         AveragedPerceptronTagger, bleu, rouge_n, rouge_l)


def test_unigram_tokenizer_learns_pieces():
    corpus = ['the cats are running', 'running cats jump', 'the dog runs',
              'cats and dogs running'] * 10
    tok = UnigramTokenizer(vocab_size=60, n_iter=6).fit(corpus)
    assert len(tok.vocab_) <= 60
    # every word is segmented into pieces that reconstruct it
    enc = tok.encode('running')
    assert ''.join(enc) == 'running'
    assert len(enc) < len('running')                     # uses multi-char pieces


def test_beam_search_escapes_greedy_trap():
    vocab = [0, 1]

    def score_fn(seq):
        if len(seq) == 0:
            return np.array([1.0, 2.0])                  # greedy prefers token 1
        if seq == [0]:
            return np.array([10.0, 10.0])                # but token 0 unlocks a big payoff
        return np.array([0.0, 0.0])                      # token 1 leads nowhere

    def total(seq):
        t, cur = 0.0, []
        for tok in seq:
            t += score_fn(cur)[tok]; cur.append(tok)
        return t

    beam = beam_search(score_fn, start=[], vocab=vocab, max_len=2, beam_width=2)
    greedy = []
    for _ in range(2):
        greedy.append(vocab[int(np.argmax(score_fn(greedy)))])
    assert total(beam) > total(greedy)                   # beam finds the better path


def test_word_movers_distance_ranks_paraphrase():
    emb = {'king': np.array([1., 0]), 'queen': np.array([0.9, 0.1]),
           'man': np.array([0., 1]), 'woman': np.array([0.1, 0.9]),
           'the': np.array([0.5, 0.5])}
    close = word_movers_distance('the king', 'the queen', emb)
    far = word_movers_distance('the king', 'the woman', emb)
    assert close < far


def test_averaged_perceptron_tagger():
    sents = [['the', 'cat', 'runs'], ['the', 'dog', 'sleeps'],
             ['a', 'cat', 'sleeps'], ['a', 'dog', 'runs']] * 10
    tags = [['D', 'N', 'V']] * 40
    apt = AveragedPerceptronTagger(epochs=15, random_state=0).fit(sents, tags)
    assert apt.predict(['the', 'dog', 'runs']) == ['D', 'N', 'V']


def test_bleu_and_rouge():
    assert bleu('the cat sat on the mat', 'the cat sat on the mat') == pytest.approx(1.0)
    assert 0 < bleu('the cat sat on the mat', 'the cat sat') < 1.0
    assert bleu('the cat sat', 'dogs run fast') == 0.0
    assert rouge_n('the cat sat', 'the cat sat', n=1) == pytest.approx(1.0)
    assert 0 < rouge_l('the cat sat on mat', 'the cat on mat') < 1.0
