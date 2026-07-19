"""Extract keyphrases from word co-occurrence, no training (Rose et al., 2010)."""
import re
from collections import defaultdict, Counter


_WORD = re.compile(r"[A-Za-z']+")


_STOPWORDS = set("""a an the and or but if then of to in on at for with as by is are
was were be been being this that these those it its i you he she we they them his
her their our your from not no do does did has have had will would can could should
than so such into over under out up down about""".split())


def rake_keywords(text, top_k=5, stopwords=None):
    """Extract keyphrases from word co-occurrence, no training (Rose et al., 2010).

    RAKE (Rapid Automatic Keyword Extraction) splits text at stopwords and
    punctuation into candidate PHRASES, then scores each word by
    ``degree / frequency`` -- degree counting how many words it co-occurs with
    inside phrases, so words that appear in longer, richer phrases score higher. A
    phrase's score is the sum of its words'. Fast, unsupervised, and surprisingly
    competitive. Returns the top-``k`` phrases with scores.
    """
    stop = _STOPWORDS if stopwords is None else set(stopwords)
    # candidate phrases are maximal runs of content words, broken by a stopword or
    # any punctuation (words and punctuation are the only tokens we keep)
    phrases = []
    cur = []
    for tok in re.findall(r"[a-z']+|[.,!?;:()\"]", text.lower()):
        if _WORD.fullmatch(tok) and tok not in stop:
            cur.append(tok)
        else:                                       # stopword or punctuation -> break
            if cur:
                phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    freq = Counter(); degree = Counter()
    for ph in phrases:
        for w in ph:
            freq[w] += 1
            degree[w] += len(ph) - 1                 # co-occurrences within phrase
    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}
    scored = [(" ".join(ph), sum(word_score[w] for w in ph)) for ph in phrases]
    # dedupe keeping the max score per phrase
    best = {}
    for p, s in scored:
        best[p] = max(best.get(p, 0), s)
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


__all__ = ["rake_keywords"]
