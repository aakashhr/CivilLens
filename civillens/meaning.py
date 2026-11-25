"""Meaning Auditor: does the rewrite still say what the writer meant?

Signals
  similarity      Sentence-BERT cosine (if installed) or TF-IDF character cosine (offline)
  retention       share of the original's content words (complaint, target, time) kept in the rewrite
  fingerprint     claim / stance / target / grievance, extracted by the LLM, or heuristically offline
  invented facts  accusations or numbers that appear only in the rewrite
"""
import re

import numpy as np

from .lexicon import ACCUSATION_WORDS, NEGATIVE_WORDS, POSITIVE_WORDS, STOPWORDS, TOXIC_LEXICON
from .masking import strip_toxic

_TOXIC_WORDS = {w for ws in TOXIC_LEXICON.values() for p in ws for w in p.split()}
_WORD = re.compile(r"[a-z0-9']+")


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_words(text):
    return {_stem(w) for w in _WORD.findall(str(text).lower())
            if w not in STOPWORDS and w not in _TOXIC_WORDS and len(w) > 1}


_RESOLVED = re.compile(r"\b(no longer (a |an )?(problem|issue)|not a problem|not an issue|no problems?|no complaints?|"
                       r"never late|no delays?)\b")


def stance(text):
    low = _RESOLVED.sub(" resolved ", str(text).lower())
    words = _WORD.findall(low)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    pos = sum(w in POSITIVE_WORDS for w in words)
    # "not good", "never fixed" style negation of a positive word counts as negative
    for i, w in enumerate(words):
        if w in POSITIVE_WORDS and i > 0 and words[i - 1] in {"not", "never", "no", "nothing"}:
            pos -= 1
    if neg > pos:
        return "against"
    if pos > neg:
        return "for"
    return "neutral"


def target(text):
    m = re.search(r"\bthe ((?:[a-z-]+ ){0,2}(?:officials|corporation|councillor|board|department|authority|staff|"
                  r"administration|team|contractor|police|members|bank|company|government|reporter))\b", str(text).lower())
    return m.group(0) if m else ""


def numbers(text):
    return set(re.findall(r"\b\d+\b", str(text)))


class Embedder:
    def __init__(self, settings, sim_vectorizer):
        self.tfidf = sim_vectorizer
        self.sbert = None
        self.backend = "TF-IDF character cosine"
        if settings.embedding_backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert = SentenceTransformer(settings.sbert_model)
                self.backend = "Sentence-BERT (all-MiniLM-L6-v2)"
            except Exception:
                pass

    def similarity(self, a, b):
        if not str(a).strip() or not str(b).strip():
            return 0.0
        if self.sbert is not None:
            e = self.sbert.encode([a, b], normalize_embeddings=True)
            return float(np.clip(np.dot(e[0], e[1]), 0, 1))
        m = self.tfidf.transform([a, b])
        return float((m[0] @ m[1].T).toarray()[0, 0])


def heuristic_fingerprint(text):
    words = content_words(text)
    return {"claim": text, "stance": stance(text), "target": target(text),
            "grievance": ", ".join(sorted(words - content_words(target(text)))[:6])}


def analyse(original, rewrite, embedder, llm=None, masked_original=None):
    """Return meaning signals and a Meaning Preservation Score (MPS) in [0, 1]."""
    clean_orig = strip_toxic(original)
    sim = embedder.similarity(clean_orig, rewrite)
    ow, rw = content_words(clean_orig), content_words(rewrite)
    retention = len(ow & rw) / len(ow) if ow else 1.0
    fp_o, fp_r = heuristic_fingerprint(clean_orig), heuristic_fingerprint(rewrite)
    tgt_o = content_words(fp_o["target"])
    target_kept = (not tgt_o) or len(tgt_o & rw) / len(tgt_o) >= 0.5
    stance_same = fp_o["stance"] == fp_r["stance"] or fp_o["stance"] == "neutral"
    stance_flipped = fp_o["stance"] == "against" and fp_r["stance"] == "for"
    new_acc = sorted((set(_WORD.findall(str(rewrite).lower())) & ACCUSATION_WORDS)
                     - set(_WORD.findall(str(original).lower())))
    new_nums = sorted(numbers(rewrite) - numbers(original))
    invented = new_acc + new_nums
    result = {
        "clean_original": clean_orig, "similarity": round(sim, 4), "retention": round(retention, 4),
        "fingerprint": {"original": fp_o, "rewrite": fp_r,
                        "matches": {"claim": retention >= 0.6, "stance": stance_same, "target": target_kept,
                                    "grievance": retention >= 0.5}},
        "stance_flipped": stance_flipped, "invented_facts": invented,
        "sarcasm_llm": False, "hidden_toxicity_llm": False, "intent_recovered": None,
        "method": "heuristic", "llm_notes": "",
    }
    if llm is not None and getattr(llm, "available", False):
        from .llm import ANALYSIS_SYSTEM, LLMError
        try:
            a = llm.chat_json(ANALYSIS_SYSTEM, f"ORIGINAL (masked): {masked_original or original}\nREWRITE: {rewrite}")
            m = {"claim": bool(a.get("same_claim")), "stance": bool(a.get("same_stance")),
                 "target": bool(a.get("same_target")), "grievance": bool(a.get("same_grievance"))}
            result["fingerprint"] = {"original": a.get("original", fp_o), "rewrite": a.get("rewrite", fp_r), "matches": m}
            o_st = str(a.get("original", {}).get("stance", "")).lower()
            r_st = str(a.get("rewrite", {}).get("stance", "")).lower()
            result["stance_flipped"] = stance_flipped or (o_st == "against" and r_st == "for")
            llm_new = [str(x) for x in (a.get("new_facts") or []) if str(x).strip()]
            result["invented_facts"] = sorted(set(invented) | set(llm_new))
            result["sarcasm_llm"] = bool(a.get("sarcasm_or_mockery"))
            result["hidden_toxicity_llm"] = bool(a.get("hidden_toxicity"))
            result["intent_recovered"] = a.get("intent_recovered")
            result["llm_notes"] = str(a.get("notes", ""))
            result["method"] = f"LLM fingerprint ({llm.label}) + embeddings"
        except LLMError:
            result["method"] = "heuristic (LLM unavailable)"
    m = result["fingerprint"]["matches"]
    fp_score = sum(m.values()) / 4
    if result["method"].startswith("LLM"):
        intent = 1.0 if result["intent_recovered"] in (True, None) else 0.0
        mps = 0.35 * fp_score + 0.25 * sim + 0.25 * retention + 0.15 * intent
    else:
        mps = 0.30 * fp_score + 0.30 * sim + 0.40 * retention
    if result["stance_flipped"]:
        mps *= 0.5
    result["fingerprint_score"] = round(fp_score, 4)
    result["mps"] = round(float(np.clip(mps, 0, 1)), 4)
    return result
