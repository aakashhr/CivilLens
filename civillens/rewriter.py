"""Rewrites a toxic comment into a civil one.

Primary: LLM (Groq / Gemini / Ollama) given the MASKED comment.
Fallback: rule-based detoxifier that drops abusive sentences and words but keeps the complaint.
"""
import re

from .lexicon import NEGATIVE_WORDS
from .masking import TAG_RE, find_sarcasm, mask, tidy

_SENT = re.compile(r"(?<=[.!?])\s+")
REQUEST_VERBS = {"fix", "repair", "resolve", "clean", "restore", "send", "help", "release", "update", "check",
                 "reply", "respond", "refund", "open", "start", "finish", "act", "do", "publish", "replace"}


def rule_based_rewrite(text):
    masked, spans = mask(text)
    kept = []
    for sent in _SENT.split(masked.strip()):
        if not sent.strip():
            continue
        words = set(re.findall(r"[a-z']+", sent.lower()))
        complaint = bool(words & NEGATIVE_WORDS) or bool(words & REQUEST_VERBS) or "?" in sent
        has_tag = bool(TAG_RE.search(sent))
        cues = find_sarcasm(sent)
        if (has_tag or cues) and not complaint:
            continue  # abusive or mocking sentence with no complaint inside: drop it
        for cue in cues:
            sent = re.sub(re.escape(cue), "", sent, flags=re.I)
        # remove "<subject> is/are [TAG]" fragments and dangling words left by removed insults
        sent = re.sub(r"(,\s*)?(just\s+)?\[(HARASSMENT|THREAT)\]\s+and\s+", ", please ", sent)
        sent = re.sub(r"(,\s*)?(just\s+)?\[(HARASSMENT|THREAT)\]\s*", " ", sent)
        sent = re.sub(r"\b(is|are|was|were)\s+(a\s+)?(total\s+|complete\s+|absolute\s+|utter\s+|bunch of\s+)?(\[[A-Z]+\]\s*)+", "", sent)
        sent = re.sub(r"\bwhat the\s+\[[A-Z]+\]\??", "", sent, flags=re.I)
        sent = TAG_RE.sub("", sent)
        sent = re.sub(r"\b(honestly|wow|oh|just)\b,?", "", sent, flags=re.I)
        sent = re.sub(r"^\W*(and|but|so)\b\s*", "", tidy(sent), flags=re.I)
        sent = tidy(sent)
        if len(re.findall(r"[A-Za-z]+", sent)) >= 3:
            kept.append(sent if sent[-1] in ".!?" else sent + ".")
    out = tidy(" ".join(kept))
    if not out:
        out = "I would like this issue to be looked into."
    return out


def rewrite(text, llm=None):
    """Returns {rewrite, method, removed}."""
    masked, spans = mask(text)
    if llm is not None and getattr(llm, "available", False):
        from .llm import REWRITE_SYSTEM, LLMError
        try:
            r = llm.chat_json(REWRITE_SYSTEM, f"Comment: {masked}")
            out = tidy(str(r.get("rewrite", "")).strip())  # scored raw; masked only for display
            if out:
                return {"rewrite": out, "method": f"LLM ({llm.label})",
                        "removed": [str(x) for x in (r.get("removed") or [])]}
        except LLMError:
            pass
    out = rule_based_rewrite(text)
    removed = [f"{s['category']} phrase" for s in spans] + ["sarcasm cue" for _ in find_sarcasm(text)]
    return {"rewrite": out, "method": "rule-based detoxifier (offline)", "removed": removed}
