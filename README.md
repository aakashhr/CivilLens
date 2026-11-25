# CivilLens – Safe Public Comment Rewrite Reviewer (PS 215)

Remove the abuse, keep the point. CivilLens rewrites toxic public comments and checks that each rewrite is
**both safe and faithful** to what the writer meant.

Team CS_12 · Generative AI Hackathon · Alliance School of Advanced Computing

## Modules
| Module | What it does |
| --- | --- |
| **1 · Rewrite a comment** | Paste one comment → masked original, polite rewrite, toxicity before/after (%), toxicity by category, meaning preserved (%), CFI score, Accept / Revise / Reject verdict with reasons and cited rules, Red-Team checks, meaning fingerprint. |
| **2 · Batch review (CSV)** | Upload a CSV (or use the 100-comment sample) → every row is reviewed (empty rewrites are generated automatically) → summary metrics, charts, filterable results table, per-comment drill-down, CSV download. |

## New in Review III
* **LLM integration:** model dropdown, *Test connection* button, automatic switch to a working model if one is retired (HTTP 404), *Save key* to `.env`, LLM response cache.
* **Performance:** per-stage timing chart for every review, batch speed (ms per comment, comments/second), `compare_llm.py` for LLM vs offline results.
* **Deployment:** `Dockerfile` (with health check), Streamlit Community Cloud secrets template, `DEPLOY.md`.
* **Compare mode:** Module 1 can show the LLM and offline results side by side.

## Quick start
```bash
# Windows: double-click run_windows.bat      macOS / Linux: ./run_mac_linux.sh
# or manually:
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python train.py                   # trains the toxicity model (~5 seconds)
streamlit run app.py
```
Open http://localhost:8501.

### Add an LLM (recommended for the best rewrites)
Copy `.env.example` to `.env` and add **one** of:
* `GROQ_API_KEY` – free key from console.groq.com (default model llama-3.1-8b-instant)
* `GEMINI_API_KEY` – free key from aistudio.google.com (default model gemini-3.8-flash)
* Ollama running locally (`ollama pull llama3.1`) – no key, no internet

You can also pick the provider and paste a key in the app's sidebar. **Without any key the app still works**
using the offline models (rule-based rewriter + heuristic meaning fingerprint).

If a default model name is retired by the provider, change `GROQ_MODEL` / `GEMINI_MODEL` in `.env` or in the sidebar.

## Project structure
```
app.py                    Streamlit UI (Module 1 + Module 2)
civillens/
  masking.py              Safe Masking Layer + output firewall
  toxicity.py             Toxicity Auditor (TF-IDF + Logistic Regression, lexicon floor, optional Detoxify)
  rewriter.py             LLM rewriter (masked input) + offline rule-based detoxifier
  meaning.py              Meaning Auditor (LLM fingerprint, embeddings, retention, stance, invented facts)
  redteam.py              Red-Team Agent (sarcasm, over-cleaning, stance flip, invented facts, identity-swap fairness test)
  guidelines.py           Guidelines RAG (FAISS + SBERT if installed, else TF-IDF index)
  judge.py                Policy-Grounded Judge (CFI score, verdict, reasons, LLM explanation)
  pipeline.py             End-to-end pipeline (single + batch)
  llm.py                  Groq / Gemini / Ollama client and prompts
data/
  generate_dataset.py     builds the datasets below
  train_toxicity.csv      4,800 labelled comments (6 classes)
  rewrite_pairs.csv       600 comment + rewrite pairs with expected verdicts
  input_comments_100.csv  100-comment input file for Module 2
  handwritten_test.csv    40 hand-written comments (out-of-template test)
  guidelines.md           11 community rules used by the judge (RAG)
train.py                  trains the toxicity model
compare_llm.py            LLM vs offline comparison (needs a key)
Dockerfile, DEPLOY.md     deployment
evaluate.py               evaluates the full reviewer, writes reports/
reports/                  metrics and result CSVs
```

## Results (offline mode, no LLM)
| Test | Result |
| --- | --- |
| Reviewer verdict accuracy, 600 labelled pairs | 97.2% |
| Bad rewrites caught (not accepted) | 99.7% |
| Good rewrites wrongly rejected | 0% |
| Batch review on the 100-comment file (75 labelled rows) | 98.7% agreement |
| Rewriter: average toxicity before → after (85 toxic comments) | 99% → 6% |
| Rewriter: meaning preserved | 96% |
| Toxicity scorer on 40 hand-written comments | 95% accuracy |

The datasets are synthetic, so in-template scores are optimistic; the hand-written test is the more realistic
figure. Run `python evaluate.py --llm` to evaluate with your LLM.
