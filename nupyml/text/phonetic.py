"""Phonetic encoders: map words to how they SOUND so spellings collide."""
import re


def soundex(word):
    """Encode a name by how it SOUNDS, so spellings collide (Russell, 1918).

    To match names despite spelling variation ("Robert"/"Rupert"), Soundex keeps the
    first letter and maps the rest to digits by consonant CLASS (labials together,
    gutturals together...), dropping vowels and repeats. Similar-sounding names get the
    same four-character code, which is why it has been the standard for phonetic name
    matching in census and record linkage for a century. Returns the 4-character code.
    """
    word = re.sub(r"[^A-Za-z]", "", word).upper()
    if not word:
        return "0000"
    codes = {**dict.fromkeys("BFPV", "1"), **dict.fromkeys("CGJKQSXZ", "2"),
             **dict.fromkeys("DT", "3"), **dict.fromkeys("L", "4"),
             **dict.fromkeys("MN", "5"), **dict.fromkeys("R", "6")}
    first = word[0]
    prev = codes.get(first, "")
    out = ""
    for ch in word[1:]:
        d = codes.get(ch, "")
        if d and d != prev:                                 # skip adjacent duplicates
            out += d
        if ch not in "HW":                                  # H, W don't reset the run
            prev = d
    return (first + out + "000")[:4]


def metaphone(word):
    """Phonetic code that handles English spelling rules (Philips, 1990).

    Soundex is crude -- it ignores that "ph" sounds like "f" and "gh" is often silent.
    Metaphone applies a set of English pronunciation RULES to produce a variable-length
    consonant key that captures sound far better, so it matches homophones Soundex
    misses. This is a compact version of the core transformations. Returns the phonetic
    key (a consonant string).
    """
    w = re.sub(r"[^A-Za-z]", "", word).upper()
    if not w:
        return ""
    w = re.sub(r"([^C])\1", r"\1", w)                       # drop doubled letters
    w = w.replace("PH", "F").replace("GH", "").replace("TH", "0")    # gh is silent
    w = w.replace("SH", "X").replace("CH", "X")
    out = []
    vowels = "AEIOU"
    for i, ch in enumerate(w):
        if ch in vowels:
            if i == 0:
                out.append(ch)                              # keep only a leading vowel
        elif not out or out[-1] != ch:
            if ch == "C":
                out.append("S" if i + 1 < len(w) and w[i + 1] in "IEY" else "K")
            elif ch == "Q":
                out.append("K")
            elif ch == "V":
                out.append("F")
            elif ch == "Z":
                out.append("S")
            elif ch in "WY" and (i + 1 >= len(w) or w[i + 1] not in vowels):
                continue
            else:
                out.append(ch)
    return "".join(out)


__all__ = ["soundex", "metaphone"]
