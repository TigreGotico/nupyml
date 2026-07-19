"""M3: language v5 -- pLSA, Aho-Corasick, phonetic codes, YAKE, language ID,
Good-Turing.

pLSA separates a two-topic corpus; Aho-Corasick finds every planted pattern in one
pass; Soundex/Metaphone collide homophones; YAKE surfaces the salient phrase;
the character n-gram detector identifies the language; Good-Turing reserves the
right missing mass and renormalises.
"""
import numpy as np

from nupyml.text import (PLSA, AhoCorasick, soundex, metaphone, yake_keywords,
                        LanguageDetector, good_turing_smoothing)


def test_plsa_separates_two_topics():
    rng = np.random.RandomState(0)
    docs = [rng.multinomial(30, [.3, .3, .3, .03, .03, .01]) for _ in range(20)]
    docs += [rng.multinomial(30, [.03, .03, .01, .3, .3, .3]) for _ in range(20)]
    X = np.array(docs, float)
    pl = PLSA(n_topics=2, random_state=0).fit(X)
    dt = pl.doc_topic_
    # the two document groups load on different topics
    assert abs(dt[:20, 0].mean() - dt[20:, 0].mean()) > 0.5
    assert np.allclose(dt.sum(axis=1), 1)


def test_aho_corasick_finds_all_patterns():
    ac = AhoCorasick(["he", "she", "his", "hers"])
    hits = ac.search("ushers")
    found = {p for _, p in hits}
    assert "she" in found and "he" in found and "hers" in found
    # positions are correct
    assert (1, "she") in hits and (2, "hers") in hits


def test_phonetic_codes_collide_homophones():
    assert soundex("Robert") == soundex("Rupert")          # classic Soundex example
    assert soundex("Ashcraft") == "A261"
    assert metaphone("night") == metaphone("nite")
    assert metaphone("phone") == metaphone("fone")


def test_yake_surfaces_salient_phrase():
    text = ("Machine learning is powerful. Machine learning models learn patterns. "
            "Deep learning is a kind of machine learning.")
    kws = [k for k, _ in yake_keywords(text, top_k=6, ngram=2)]
    assert any("machine" in k or "learning" in k for k in kws[:3])


def test_language_detector_identifies_language():
    en = "the quick brown fox jumps over the lazy dog and the cat"
    pt = "o rapido gato preto salta sobre o cao e a raposa que corre"
    ld = LanguageDetector(n=3).fit([en, pt], ["en", "pt"])
    assert ld.predict("the dog and the fox") == "en"
    assert ld.predict("o gato e o cao") == "pt"


def test_good_turing_reserves_missing_mass():
    p0, sm = good_turing_smoothing({"a": 3, "b": 1, "c": 1, "d": 2, "e": 1})
    # three singletons out of 8 tokens -> missing mass 3/8
    assert abs(p0 - 3 / 8) < 1e-9
    assert abs(sum(sm.values()) + p0 - 1.0) < 1e-9         # a proper distribution
