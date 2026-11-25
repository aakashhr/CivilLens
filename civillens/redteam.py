"""Red-Team Agent: looks for problems the other two auditors can miss."""
import re

from .lexicon import IDENTITY_SWAPS, IDENTITY_TERMS
from .masking import TAG_RE


def _identity_terms(text):
    low = str(text).lower()
    return [t for t in IDENTITY_TERMS if re.search(r"\b" + re.escape(t) + r"\b", low)]


def fairness_test(text, scorer, threshold=0.20):
    """Swap identity words and re-score. A big change means the scorer is biased for this text."""
    terms = _identity_terms(text)
    if not terms:
        return {"tested": False, "terms": [], "max_shift": 0.0, "biased": False}
    base = scorer.score(text)["overall"]
    variants = []
    for t in terms:
        swapped = re.sub(r"\b" + re.escape(t) + r"\b", IDENTITY_SWAPS.get(t, "person"), text, flags=re.I)
        variants.append(swapped)
    scores = [s["overall"] for s in scorer.score_many(variants)]
    shift = max(abs(base - s) for s in scores)
    return {"tested": True, "terms": terms, "max_shift": round(shift, 4), "biased": shift > threshold}


def run_checks(original, rewrite, tox_before, tox_after, meaning, scorer, settings):
    """Return (checks, flags). checks = [{name, status: pass|flag, detail}]."""
    checks, flags = [], []

    def add(name, ok, detail, flag=None):
        checks.append({"check": name, "status": "pass" if ok else "flag", "detail": detail})
        if not ok and flag:
            flags.append(flag)

    words = re.findall(r"[A-Za-z]+", str(rewrite))
    fluent = len(words) >= 3 and not TAG_RE.search(str(rewrite)) and not re.search(r"([!?.])\1{2,}", str(rewrite))
    add("Fluent, complete rewrite", fluent, "Readable sentence" if fluent else "Rewrite is too short or broken", "fluency")

    surviving = tox_after["spans"]
    cats = sorted({s["category"] for s in surviving})
    add("No toxic words left", not surviving,
        "All toxic words removed" if not surviving else f"Still contains: {', '.join(c + ' words' for c in cats)}",
        None)
    for c in cats:
        flags.append({"insult": "residual_toxicity"}.get(c, c))
    resid = tox_after["overall"] < settings.residual_toxicity
    add("Residual toxicity below limit", resid, f"Toxicity after rewrite: {tox_after['overall']:.0%}",
        "residual_toxicity" if not surviving and not resid else None)

    sarcasm = bool(tox_after["sarcasm_cues"]) or meaning.get("sarcasm_llm") or meaning.get("hidden_toxicity_llm")
    add("No sarcasm or mockery", not sarcasm,
        "No mocking praise found" if not sarcasm else "Mocking or sarcastic phrasing detected", "sarcasm")

    over = meaning["retention"] < 0.35 and meaning["similarity"] < 0.55
    add("Complaint kept (not over-cleaned)", not over,
        f"{meaning['retention']:.0%} of the original's key words kept", "over_cleaned")

    add("Stance unchanged", not meaning["stance_flipped"],
        "Same stance as original" if not meaning["stance_flipped"] else "Complaint turned into praise", "stance_flipped")

    inv = meaning["invented_facts"]
    add("No invented facts", not inv, "Nothing new added" if not inv else f"New content: {', '.join(map(str, inv[:4]))}",
        "invented_facts")

    fair = fairness_test(original, scorer)
    lost = [t for t in fair["terms"] if t not in _identity_terms(rewrite)] if fair["tested"] else []
    ok = not fair["biased"] and not lost
    detail = ("No identity words in this comment" if not fair["tested"] else
              f"Identity words: {', '.join(fair['terms'])}; score shift on swap {fair['max_shift']:.0%}"
              + (f"; removed from rewrite: {', '.join(lost)}" if lost else ""))
    add("Identity fairness test", ok, detail, "identity" if lost else None)
    return checks, flags, fair
