"""Evaluate the full CivilLens reviewer.

1. rewrite_pairs.csv (600 pairs with known rewrite quality): verdict accuracy and catch rate per failure type.
2. input_comments_100.csv: runs Batch Review exactly as the app does and saves the results.
3. Rewriter check: rewrites every toxic comment in input_comments_100.csv and measures toxicity drop.

Usage: python evaluate.py            (offline, no LLM)
       python evaluate.py --llm      (uses the LLM configured in .env; slower)
"""
import json
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from civillens.config import DATA_DIR, REPORT_DIR
from civillens.pipeline import CivilLens
from civillens.rewriter import rewrite

use_llm = "--llm" in sys.argv
REPORT_DIR.mkdir(exist_ok=True)
cl = CivilLens()
print("Components:", cl.components)

# 1. Reviewer accuracy on labelled pairs
pairs = pd.read_csv(DATA_DIR / "rewrite_pairs.csv")
res = cl.review_batch(pairs, auto_rewrite=False, use_llm=use_llm)
res["expected_verdict"] = pairs["expected_verdict"].values
res["rewrite_type"] = pairs["rewrite_type"].values
labels = ["Accept", "Revise", "Reject"]
acc = accuracy_score(res.expected_verdict, res.verdict)
per_type = (res.assign(correct=res.expected_verdict == res.verdict)
            .groupby("rewrite_type")["correct"].mean().round(4).to_dict())
caught = res[res.rewrite_type != "good"]
catch_rate = float((caught.verdict != "Accept").mean())
false_alarm = float((res[res.rewrite_type == "good"].verdict != "Accept").mean())
res.to_csv(REPORT_DIR / "evaluation_rewrite_pairs.csv", index=False)

# 2. Batch review on the 100-comment input file
inp = pd.read_csv(DATA_DIR / "input_comments_100.csv")
batch = cl.review_batch(inp, auto_rewrite=True, use_llm=use_llm)
batch.to_csv(REPORT_DIR / "input_100_results.csv", index=False)
lab = batch[batch.expected_verdict.fillna("") != ""]
batch_acc = accuracy_score(lab.expected_verdict, lab.verdict)

# 3. Rewriter: toxicity before vs after on toxic comments
tox = inp[inp.category != "none"]
rw = [rewrite(t, cl.llm if use_llm else None)["rewrite"] for t in tox.original_comment]
sb = cl.scorer.score_many(list(tox.original_comment) + rw)
n = len(tox)
before = [s["overall"] for s in sb[:n]]
after = [s["overall"] for s in sb[n:]]
rew_reports = [cl.review(o, r, use_llm=False) for o, r in zip(tox.original_comment, rw)]

# 4. Toxicity scorer on hand-written comments that do NOT come from the generator templates
hw = pd.read_csv(DATA_DIR / "handwritten_test.csv")
hw_pred = [int(s["overall"] >= 0.5) for s in cl.scorer.score_many(list(hw.text))]
hw_model_only = [int(p) for p in (cl.scorer.clf.predict(hw.text) != "none")]
from sklearn.metrics import f1_score
hw_res = {"comments": int(len(hw)), "accuracy": round(accuracy_score(hw.is_toxic, hw_pred), 4),
          "f1": round(f1_score(hw.is_toxic, hw_pred), 4),
          "accuracy_model_only_no_lexicon": round(accuracy_score(hw.is_toxic, hw_model_only), 4),
          "missed": [t for t, y, p in zip(hw.text, hw.is_toxic, hw_pred) if y == 1 and p == 0],
          "false_alarms": [t for t, y, p in zip(hw.text, hw.is_toxic, hw_pred) if y == 0 and p == 1]}

metrics = {
    "toxicity_on_handwritten_test": hw_res,
    "mode": "LLM" if use_llm else "offline",
    "components": cl.components,
    "reviewer_on_600_pairs": {
        "verdict_accuracy": round(acc, 4), "catch_rate_bad_rewrites": round(catch_rate, 4),
        "false_alarm_rate_good_rewrites": round(false_alarm, 4), "accuracy_by_rewrite_type": per_type,
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(res.expected_verdict, res.verdict, labels=labels).tolist()},
        "verdict_counts": res.verdict.value_counts().to_dict(),
    },
    "batch_review_input_100": {
        "rows": int(len(batch)), "verdict_counts": batch.verdict.value_counts().to_dict(),
        "verdict_accuracy_on_labelled_rows": round(batch_acc, 4), "labelled_rows": int(len(lab)),
        "avg_toxicity_before": round(float(batch.toxicity_before.mean()), 4),
        "avg_toxicity_after": round(float(batch.toxicity_after.mean()), 4),
        "avg_meaning_preserved": round(float(batch.meaning_preserved.mean()), 4),
    },
    "rewriter_on_toxic_comments": {
        "comments": n, "avg_toxicity_before": round(sum(before) / n, 4), "avg_toxicity_after": round(sum(after) / n, 4),
        "share_below_30pct_after": round(sum(a < 0.3 for a in after) / n, 4),
        "avg_meaning_preserved": round(sum(r["mps"] for r in rew_reports) / n, 4),
        "accepted_share": round(sum(r["verdict"] == "Accept" for r in rew_reports) / n, 4),
    },
}
pd.DataFrame({"original_masked": [r["original_masked"] for r in rew_reports], "rewrite": rw,
              "toxicity_before": before, "toxicity_after": after,
              "meaning_preserved": [r["mps"] for r in rew_reports], "verdict": [r["verdict"] for r in rew_reports]}
             ).to_csv(REPORT_DIR / "rewriter_results.csv", index=False)
(REPORT_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: v for k, v in metrics.items() if k != "components"}, indent=2))
