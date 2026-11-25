"""Safe Masking Layer: replaces toxic words with category tags before anything is shown or sent to an LLM."""
import re

from .lexicon import SARCASM_CUES, TOXIC_LEXICON

TAGS = {"insult": "[INSULT]", "profanity": "[PROFANITY]", "harassment": "[HARASSMENT]", "threat": "[THREAT]"}
TAG_RE = re.compile(r"\[(INSULT|PROFANITY|HARASSMENT|THREAT)\]")

_INDIC = "\u0900-\u097F\u0C80-\u0CFF"


def _pattern(phrase):
    if re.search(f"[{_INDIC}]", phrase):  # Indic scripts: allow inflected endings (e.g. ಮೂರ್ಖರು)
        return re.compile(f"(?<![\\w{_INDIC}])" + re.escape(phrase) + f"[{_INDIC}]*")
    return re.compile(f"(?<![\\w{_INDIC}])" + re.escape(phrase) + f"(?![\\w{_INDIC}])", re.I)


_PATTERNS = sorted(
    [(cat, phrase, _pattern(phrase)) for cat, phrases in TOXIC_LEXICON.items() for phrase in phrases],
    key=lambda x: len(x[1]), reverse=True)
_SARCASM = [(c, re.compile(re.escape(c), re.I)) for c in sorted(SARCASM_CUES, key=len, reverse=True)]


def find_spans(text):
    """Return non-overlapping toxic spans: [{phrase, category, start, end}]."""
    spans, taken = [], []
    for cat, phrase, pat in _PATTERNS:
        for m in pat.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append({"phrase": m.group(0), "category": cat, "start": m.start(), "end": m.end()})
    return sorted(spans, key=lambda s: s["start"])


def find_sarcasm(text):
    found, taken = [], []
    for cue, pat in _SARCASM:
        for m in pat.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append(cue)
    return found


def mask(text):
    """Replace toxic spans with tags. Returns (masked_text, spans)."""
    text = "" if text is None else str(text)
    spans = find_spans(text)
    out, last = [], 0
    for s in spans:
        out.append(text[last:s["start"]])
        out.append(TAGS[s["category"]])
        last = s["end"]
    out.append(text[last:])
    return "".join(out), spans


def firewall(text):
    """Output firewall: re-mask any generated text before it is displayed."""
    return mask(text)[0]


def tidy(text):
    """Clean spacing and punctuation after words were removed."""
    t = TAG_RE.sub("", text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"([,;:])\s*([.!?])", r"\2", t)
    t = re.sub(r"([.!?])[,;:]+", r"\1", t)
    t = re.sub(r"^[\s,.;:!?-]+", "", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"([!?.]){2,}", r"\1", t)
    t = t.strip()
    # capitalise each sentence
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t


def strip_toxic(text):
    """Original text with toxic words and sarcasm cues removed (used to measure meaning)."""
    masked, _ = mask(text)
    for cue, pat in _SARCASM:
        masked = pat.sub("", masked)
    return tidy(masked)
