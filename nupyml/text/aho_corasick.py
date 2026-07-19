"""Aho-Corasick: match many patterns in a single pass with a failure automaton."""
from collections import deque


class AhoCorasick:
    """Match MANY patterns in one pass with an automaton (Aho & Corasick, 1975).

    Searching a text for each of a thousand keywords separately is a thousand scans.
    Aho-Corasick builds all the patterns into a single trie, then adds FAILURE links
    (where to fall back when a match breaks, like KMP generalised to many strings), so
    a single left-to-right pass over the text reports every occurrence of every
    pattern. It is the workhorse behind virus scanners and dictionary taggers. Build
    with ``add``/``build`` (or pass patterns to the constructor), then ``search``.
    """

    def __init__(self, patterns=None):
        self.goto = [{}]                                    # trie transitions
        self.out = [set()]                                  # patterns ending at node
        self.fail = [0]
        self._patterns = []
        for p in patterns or []:
            self.add(p)
        if patterns:
            self.build()

    def add(self, pattern):
        node = 0
        for ch in pattern:
            if ch not in self.goto[node]:
                self.goto.append({}); self.out.append(set()); self.fail.append(0)
                self.goto[node][ch] = len(self.goto) - 1
            node = self.goto[node][ch]
        self.out[node].add(pattern)
        self._patterns.append(pattern)
        return self

    def build(self):
        q = deque()
        for ch, nxt in self.goto[0].items():
            self.fail[nxt] = 0; q.append(nxt)
        while q:                                            # BFS to set failure links
            node = q.popleft()
            for ch, nxt in self.goto[node].items():
                q.append(nxt)
                f = self.fail[node]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.goto[f].get(ch, 0) if f or ch in self.goto[0] else 0
                self.out[nxt] |= self.out[self.fail[nxt]]   # inherit suffix matches
        return self

    def search(self, text):
        node = 0; hits = []
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pat in self.out[node]:
                hits.append((i - len(pat) + 1, pat))
        return sorted(hits)


__all__ = ["AhoCorasick"]
