"""LLM vs offline comparison.

Runs the same comments through CivilLens twice - once with the configured LLM and once fully offline -
and compares verdict accuracy, rewrite quality and speed.

Usage: python compare_llm.py            (20 labelled pairs + 10 rewrites; stays within free-tier limits)
       python compare_llm.py 50         (more pairs)
Needs an LLM key in .env (GEMINI_API_KEY or GROQ_API_KEY).
"""
import json
import sys
import time

import pandas as pd

from civillens.config import DATA_DIR, REPORT_DIR
from civillens.pipeline import CivilLens

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
cl = CivilLens()
if not cl.llm.available:
    sys.exit("No LLM configured. Add GEMINI_API_KEY or GROQ_API_KEY to .env first.")
ok, msg = cl.llm.test_connection()
print(msg)
if not ok:
    sys.exit(1)

pairs = pd.read_csv(DATA_DIR / "rewrite_pairs.csv")
sample = pairs.groupby("rewrite_type", group_keys=False).apply(
    lambda g: g.sample(min(len(g), max(1, n // 6)), random_state=1)).reset_index(drop=True)
toxic = pd.read_csv(DATA_DIR / "input_comments_100.csv")
toxic = toxic[toxic.category != "none"].head(max(5, n // 2))

out = {}
for mode, use_llm in (("offline", False), ("llm", True)):
    t0 = time.time()
    verdicts = []
    for o, r in zip(sample.original_comment, sample.candidate_rewrite):
        verdicts.append(cl.review(o, r, use_llm=use_llm)["verdict"])
    review_time = time.time() - t0
    t1 = time.time()
    reps = [cl.rewrite_and_review(o, use_llm=use_llm) for o in toxic.original_comment]
    rewrite_time = time.time() - t1
    acc = float((pd.Series(verdicts) == sample.expected_verdict).mean())
    out[mode] = {
        "reviewer_accuracy": round(acc, 4),
        "reviewer_seconds_per_pair": round(review_time / len(sample), 3),
        "rewriter_avg_toxicity_after": round(sum(r["tox_after"] for r in reps) / len(reps), 4),
        "rewriter_avg_meaning_preserved": round(sum(r["mps"] for r in reps) / len(reps), 4),
        "rewriter_accept_rate": round(sum(r["verdict"] == "Accept" for r in reps) / len(reps), 4),
        "rewriter_seconds_per_comment": round(rewrite_time / len(reps), 3),
        "examples": [{"original_masked": r["original_masked"], "rewrite": r["rewrite_display"], "verdict": r["verdict"]}
                     for r in reps[:5]],
    }
    print(mode, {k: v for k, v in out[mode].items() if k != "examples"})
out["model"] = f"{cl.llm.provider} {cl.llm.model}"
out["pairs"] = int(len(sample)); out["rewrites"] = int(len(toxic))
REPORT_DIR.mkdir(exist_ok=True)
(REPORT_DIR / "llm_vs_offline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("Saved reports/llm_vs_offline.json")
