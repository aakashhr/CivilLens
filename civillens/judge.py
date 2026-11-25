"""Policy-Grounded Judge: combines all signals into the Civility-Fidelity Index (CFI) and a verdict."""

HARD_REJECT = {"stance_flipped": "the rewrite turns the complaint into praise",
               "invented_facts": "the rewrite adds facts or accusations that were not in the original",
               "threat": "a threat is still present"}
FIXABLE = {"residual_toxicity": "insulting words are still present",
           "profanity": "profanity is still present",
           "harassment": "dismissive or harassing language is still present",
           "sarcasm": "the rewrite still mocks or uses sarcasm",
           "over_cleaned": "the rewrite removed the citizen's actual complaint",
           "identity": "a respectful identity mention was removed",
           "fluency": "the rewrite is not a clear, complete sentence"}

QUERY_TEXT = {"residual_toxicity": "personal attacks insults", "profanity": "profanity swear words",
              "threat": "threats intimidation", "harassment": "harassment dismissal shut up",
              "sarcasm": "sarcasm mockery", "over_cleaned": "keep the original complaint",
              "stance_flipped": "do not change stance praise", "invented_facts": "do not add facts accusations",
              "identity": "identity groups respect", "fluency": "clear readable"}


def toxicity_reduction(before, after):
    if before >= 0.10:
        return max(0.0, min(1.0, (before - after) / before))
    return max(0.0, 1.0 - after)  # original was already clean: reward staying clean


def cfi(trs, mps, fluency_ok=True):
    if trs + mps == 0:
        return 0.0
    return round((2 * trs * mps / (trs + mps)) * (1.0 if fluency_ok else 0.6), 4)


def decide(tox_before, tox_after, meaning, flags, settings, rules_index, llm=None, masked_original="", rewrite=""):
    trs = round(toxicity_reduction(tox_before["overall"], tox_after["overall"]), 4)
    mps = meaning["mps"]
    score = cfi(trs, mps, "fluency" not in flags)
    flags = list(dict.fromkeys(flags))
    hard = [f for f in flags if f in HARD_REJECT]
    fixable = [f for f in flags if f in FIXABLE]
    if hard:
        verdict = "Reject"
    elif fixable:
        verdict = "Revise"
    elif score >= settings.accept_threshold:
        verdict = "Accept"
    elif score < settings.reject_threshold:
        verdict = "Reject"
    else:
        verdict = "Revise"

    reasons = [HARD_REJECT[f].capitalize() + "." for f in hard] + [FIXABLE[f].capitalize() + "." for f in fixable]
    if verdict == "Accept":
        reasons.append(f"Toxicity dropped from {tox_before['overall']:.0%} to {tox_after['overall']:.0%} "
                       f"and {mps:.0%} of the meaning was preserved.")
    elif not reasons:
        reasons.append(f"The CFI score of {score:.2f} is below the acceptance level of {settings.accept_threshold:.2f}.")
    query = " ".join(QUERY_TEXT[f] for f in flags) or "keep the original complaint polite no personal attacks"
    rules = rules_index.rules_for(flags, query)

    explanation = ""
    if llm is not None and getattr(llm, "available", False):
        from .llm import EXPLAIN_SYSTEM, LLMError
        from .masking import firewall
        try:
            r = llm.chat_json(EXPLAIN_SYSTEM, (
                f"Original (masked): {masked_original}\nRewrite: {rewrite}\nVerdict: {verdict}\n"
                f"Toxicity before {tox_before['overall']:.2f}, after {tox_after['overall']:.2f}; "
                f"meaning preservation {mps:.2f}; CFI {score:.2f}\nFlags: {flags or 'none'}\n"
                f"Rules: {[r['id'] + ' ' + r['title'] for r in rules]}"))
            explanation = firewall(str(r.get("explanation", "")))
        except LLMError:
            explanation = ""
    if not explanation:
        cite = ", ".join(f"{r['id']} ({r['title']})" for r in rules[:2])
        explanation = f"{verdict}: " + " ".join(reasons) + (f" Relevant rules: {cite}." if cite else "")
    return {"trs": trs, "mps": mps, "cfi": score, "verdict": verdict, "flags": flags,
            "reasons": reasons, "rules": rules, "explanation": explanation}
