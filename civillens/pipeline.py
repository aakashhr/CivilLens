"""CivilLens pipeline: Upload -> Safe Masking -> 3 agents -> Policy-Grounded Judge -> Safe Report."""
import time

import pandas as pd

from .config import Settings
from .guidelines import GuidelineIndex
from .judge import decide
from .llm import LLMClient
from .masking import firewall, mask
from .meaning import Embedder, analyse
from .redteam import run_checks
from .rewriter import rewrite as do_rewrite
from .toxicity import ToxicityScorer, load_bundle


class CivilLens:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        bundle = load_bundle()
        self.scorer = ToxicityScorer(self.settings, bundle)
        self.embedder = Embedder(self.settings, bundle["sim_vectorizer"])
        self.rules = GuidelineIndex(self.embedder)
        self.llm = LLMClient(self.settings)

    def set_llm(self, provider=None, api_key=None, model=None):
        self.llm = LLMClient(self.settings, provider=provider, api_key=api_key, model=model)
        return self.llm

    @property
    def components(self):
        return {"LLM": self.llm.label, "Toxicity": self.scorer.backend,
                "Embeddings": self.embedder.backend, "Guidelines RAG": self.rules.backend}

    # ------------------------------------------------------------------
    def review(self, original, rewrite, use_llm=True, tox_before=None, tox_after=None):
        """Review one (original, rewrite) pair and return a full report dict with per-stage timings."""
        t0 = time.time()
        T = {}
        llm = self.llm if use_llm else None

        def lap(name, start):
            T[name] = round((time.time() - start) * 1000, 1)  # milliseconds
            return time.time()

        t = time.time()
        masked, spans = mask(original)
        t = lap("Safe Masking", t)
        if tox_before is None or tox_after is None:
            tox_before, tox_after = self.scorer.score_many([original, rewrite])
        t = lap("Toxicity Auditor", t)
        meaning = analyse(original, rewrite, self.embedder, llm=llm, masked_original=masked)
        t = lap("Meaning Auditor" + (" (LLM)" if meaning["method"].startswith("LLM") else ""), t)
        checks, flags, fairness = run_checks(original, rewrite, tox_before, tox_after, meaning, self.scorer, self.settings)
        t = lap("Red-Team Agent", t)
        j = decide(tox_before, tox_after, meaning, flags, self.settings, self.rules, llm=llm,
                   masked_original=masked, rewrite=firewall(rewrite))
        t = lap("Judge + RAG" + (" (LLM)" if llm is not None and llm.available and not llm.last_error else ""), t)
        return {
            "original_masked": masked, "rewrite": rewrite, "rewrite_display": firewall(rewrite),
            "toxic_spans": [{"category": s["category"], "tag": f"[{s['category'].upper()}]"} for s in spans],
            "tox_before": tox_before["overall"], "tox_after": tox_after["overall"],
            "categories_before": tox_before["categories"], "categories_after": tox_after["categories"],
            "toxicity_reduction": round(max(0.0, tox_before["overall"] - tox_after["overall"]), 4),
            "similarity": meaning["similarity"], "retention": meaning["retention"],
            "fingerprint": meaning["fingerprint"], "meaning_method": meaning["method"],
            "intent_recovered": meaning["intent_recovered"], "llm_notes": firewall(meaning["llm_notes"]),
            "checks": checks, "fairness": fairness, **j,
            "timings_ms": T, "seconds": round(time.time() - t0, 2),
            "llm_model": f"{self.llm.provider} · {self.llm.model}" if (use_llm and self.llm.available) else "offline",
            "llm_note": self.llm.note if use_llm else "",
        }

    def rewrite_and_review(self, original, use_llm=True):
        t = time.time()
        r = do_rewrite(original, self.llm if use_llm else None)
        rw_ms = round((time.time() - t) * 1000, 1)
        report = self.review(original, r["rewrite"], use_llm=use_llm)
        name = "Rewriter (LLM)" if r["method"].startswith("LLM") else "Rewriter (offline)"
        report["timings_ms"] = {name: rw_ms, **report["timings_ms"]}
        report["seconds"] = round(report["seconds"] + rw_ms / 1000, 2)
        report.update({"rewrite_method": r["method"], "removed": r["removed"]})
        return report

    def compare(self, original):
        """Run the same comment with the LLM and fully offline, for a side-by-side comparison."""
        return {"llm": self.rewrite_and_review(original, use_llm=True),
                "offline": self.rewrite_and_review(original, use_llm=False)}

    # ------------------------------------------------------------------
    def review_batch(self, df, text_col="original_comment", rewrite_col="candidate_rewrite",
                     auto_rewrite=True, use_llm=False, progress=None):
        """Review every row of a DataFrame. Rows without a rewrite are rewritten automatically."""
        t_start = time.time()
        df = df.copy()
        if rewrite_col not in df.columns:
            df[rewrite_col] = ""
        df[rewrite_col] = df[rewrite_col].fillna("").astype(str)
        df[text_col] = df[text_col].fillna("").astype(str)
        llm = self.llm if use_llm else None
        rewrites, methods = [], []
        for i, (orig, rew) in enumerate(zip(df[text_col], df[rewrite_col])):
            if not rew.strip() and auto_rewrite:
                r = do_rewrite(orig, llm)
                rewrites.append(r["rewrite"]); methods.append("auto: " + r["method"])
            else:
                rewrites.append(rew); methods.append("provided")
        scores = self.scorer.score_many(list(df[text_col]) + rewrites)
        n = len(df)
        before, after = scores[:n], scores[n:]
        rows = []
        for i in range(n):
            rep = self.review(df[text_col].iloc[i], rewrites[i], use_llm=use_llm,
                              tox_before=before[i], tox_after=after[i])
            rows.append({
                "original_masked": rep["original_masked"], "rewrite": rep["rewrite_display"], "rewrite_raw": rewrites[i],
                "rewrite_source": methods[i],
                "toxicity_before": rep["tox_before"], "toxicity_after": rep["tox_after"],
                "toxicity_reduction": rep["trs"], "meaning_preserved": rep["mps"], "cfi": rep["cfi"],
                "verdict": rep["verdict"], "flags": ", ".join(rep["flags"]),
                "reasons": " ".join(rep["reasons"]),
                "rules": ", ".join(r["id"] for r in rep["rules"]),
                "top_category_before": max(rep["categories_before"], key=rep["categories_before"].get)
                if rep["tox_before"] >= 0.5 else "none",
            })
            if progress:
                progress(i + 1, n)
        out = pd.DataFrame(rows)
        keep = [c for c in ("comment_id", "expected_verdict", "rewrite_type") if c in df.columns]
        res = pd.concat([df[keep].reset_index(drop=True), out], axis=1)
        total = time.time() - t_start
        res.attrs["perf"] = {"total_seconds": round(total, 2), "rows": n,
                             "ms_per_comment": round(total * 1000 / max(n, 1), 1),
                             "comments_per_second": round(n / total, 1) if total > 0 else None,
                             "auto_rewritten": sum(m.startswith("auto") for m in methods)}
        return res
