"""CivilLens web app.  Run:  streamlit run app.py"""
import html

import altair as alt
import pandas as pd
import streamlit as st

from civillens.config import DATA_DIR, ROOT
from civillens.llm import CACHE_STATS, MODEL_CHOICES
from civillens.masking import firewall
from civillens.pipeline import CivilLens

st.set_page_config(page_title="CivilLens", page_icon="◐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea, select, table,
[data-testid="stSidebar"], .stTabs, h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }
.stApp { background: #FFFFFF; color: #111111; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3, h4 { color: #111111; font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #F6F6F6; border-right: 1px solid #E5E5E5; }
.cl-hero h1 { font-size: 34px; margin-bottom: 0; }
.cl-hero p { color: #6B6B6B; margin-top: 4px; font-size: 15px; }
.cl-card { border: 1px solid #E5E5E5; border-radius: 12px; padding: 16px 18px; background: #FFFFFF; height: 100%; }
.cl-card.soft { background: #F6F6F6; border-color: #EEEEEE; }
.cl-label { font-size: 11.5px; color: #6B6B6B; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 6px; font-weight: 500; }
.cl-text { font-size: 15.5px; color: #111111; line-height: 1.6; }
.cl-muted { color: #6B6B6B; font-size: 12.5px; margin-top: 8px; }
.cl-metric { border: 1px solid #E5E5E5; border-radius: 12px; padding: 12px 14px; background: #FFFFFF; }
.cl-metric .v { font-size: 24px; font-weight: 600; color: #111111; line-height: 1.2; }
.cl-metric .k { font-size: 12px; color: #6B6B6B; margin-top: 2px; }
.cl-badge { display: inline-block; padding: 3px 14px; border-radius: 999px; font-size: 14px; font-weight: 600; }
.cl-Accept { background: #111111; color: #FFFFFF; }
.cl-Revise { background: #E5E5E5; color: #111111; }
.cl-Reject { background: #FFFFFF; color: #111111; border: 1.5px solid #111111; }
.cl-check { display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px solid #F0F0F0; font-size: 14px; }
.cl-check .s { width: 46px; font-weight: 600; font-size: 12px; color: #111111; }
.cl-check .s.flag { color: #FFFFFF; background: #111111; border-radius: 4px; text-align: center; height: 20px; }
.cl-check .n { width: 230px; font-weight: 500; }
.cl-check .d { color: #6B6B6B; flex: 1; }
.cl-rule { font-size: 13.5px; padding: 6px 0; color: #333333; }
div.stButton > button, div.stDownloadButton > button { background: #111111; color: #FFFFFF; border: 1px solid #111111;
  border-radius: 8px; font-weight: 500; padding: 0.45rem 1.1rem; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background: #333333; color: #FFFFFF; border-color: #333333; }
div.stButton > button:focus { color: #FFFFFF; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #E5E5E5; }
.stTabs [data-baseweb="tab"] { font-size: 15px; padding-left: 0; padding-right: 0; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #111111 !important; }
textarea, input { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading CivilLens models...")
def get_engine():
    return CivilLens()


def esc(x):
    return html.escape(str(x))


def card(label, text, note="", soft=False):
    st.markdown(f"<div class='cl-card {'soft' if soft else ''}'><div class='cl-label'>{esc(label)}</div>"
                f"<div class='cl-text'>{esc(text)}</div>"
                + (f"<div class='cl-muted'>{esc(note)}</div>" if note else "") + "</div>", unsafe_allow_html=True)


def metric(col, label, value):
    col.markdown(f"<div class='cl-metric'><div class='v'>{esc(value)}</div><div class='k'>{esc(label)}</div></div>",
                 unsafe_allow_html=True)


def badge(verdict):
    return f"<span class='cl-badge cl-{esc(verdict)}'>{esc(verdict)}</span>"


def pct(x):
    return f"{float(x) * 100:.0f}%"


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def gray_bars(df, x, y, color=None, title=""):
    enc = dict(x=alt.X(x, title=None, axis=alt.Axis(labelAngle=0)),
               y=alt.Y(y, title=None, axis=alt.Axis(format="%", grid=True)))
    if color:
        enc["xOffset"] = alt.XOffset(color, sort=["Before", "After"])
        enc["color"] = alt.Color(color, scale=alt.Scale(domain=["Before", "After"], range=["#BDBDBD", "#111111"]),
                                 legend=alt.Legend(orient="top", title=None))
    else:
        enc["color"] = alt.value("#3A3A3A")
    return (alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(**enc)
            .properties(height=260, title=title)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EEEEEE", domainColor="#BDBDBD", labelColor="#6B6B6B", labelFont="Poppins")
            .configure_legend(labelFont="Poppins", labelColor="#333333")
            .configure_title(font="Poppins", fontSize=13, color="#111111", anchor="start", fontWeight=500))


def render_report(r, reveal_original=None):
    c1, c2 = st.columns(2)
    with c1:
        card("Original comment" + (" (unmasked)" if reveal_original else " (toxic words masked)"),
             reveal_original or r["original_masked"],
             note="Detected: " + (", ".join(sorted({s["tag"] for s in r["toxic_spans"]})) or "no toxic words"))
    with c2:
        card("Rewritten comment", r["rewrite_display"], note=r.get("rewrite_method", "Rewrite provided by user"), soft=True)
    st.write("")
    m = st.columns(6)
    metric(m[0], "Toxicity before", pct(r["tox_before"]))
    metric(m[1], "Toxicity after", pct(r["tox_after"]))
    metric(m[2], "Toxicity removed", pct(r["trs"]))
    metric(m[3], "Meaning preserved", pct(r["mps"]))
    metric(m[4], "CFI score", f"{r['cfi']:.2f}")
    m[5].markdown(f"<div class='cl-metric'><div class='v'>{badge(r['verdict'])}</div><div class='k'>Verdict</div></div>",
                  unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        cats = pd.DataFrame([{"Category": k.capitalize(), "When": w, "Score": v}
                             for w, d in (("Before", r["categories_before"]), ("After", r["categories_after"]))
                             for k, v in d.items()])
        show_chart(gray_bars(cats, "Category:N", "Score:Q", "When:N", "Toxicity by category, before vs after"))
    with right:
        rules = "".join(f"<div class='cl-rule'><b>{esc(x['id'])}</b> · {esc(x['title'])}</div>" for x in r["rules"])
        st.markdown(f"<div class='cl-card'><div class='cl-label'>Judge's verdict</div>"
                    f"<div style='margin:4px 0 10px'>{badge(r['verdict'])}</div>"
                    f"<div class='cl-text' style='font-size:14.5px'>{esc(r['explanation'])}</div>"
                    f"<div class='cl-label' style='margin-top:14px'>Community rules cited (RAG)</div>{rules}</div>",
                    unsafe_allow_html=True)
    st.markdown("#### Red-Team checks")
    rows = "".join(
        f"<div class='cl-check'><div class='s {'flag' if c['status'] == 'flag' else ''}'>"
        f"{'FLAG' if c['status'] == 'flag' else 'PASS'}</div><div class='n'>{esc(c['check'])}</div>"
        f"<div class='d'>{esc(c['detail'])}</div></div>" for c in r["checks"])
    st.markdown(f"<div class='cl-card'>{rows}</div>", unsafe_allow_html=True)
    st.markdown("#### Meaning fingerprint")
    fp = r["fingerprint"]
    show_df(pd.DataFrame([{"Field": f.capitalize(), "Original": firewall(str(fp["original"].get(f, ""))),
                           "Rewrite": firewall(str(fp["rewrite"].get(f, ""))),
                           "Match": "Yes" if fp["matches"].get(f) else "No"}
                          for f in ("claim", "stance", "target", "grievance")]))
    st.caption(f"Method: {r['meaning_method']} · semantic similarity {pct(r['similarity'])} · "
               f"key words kept {pct(r['retention'])} · reviewed in {r['seconds']} s")
    st.markdown("#### Performance")
    tm = r.get("timings_ms", {})
    if tm:
        pdf = pd.DataFrame({"Stage": list(tm), "ms": list(tm.values())})
        perf_chart = (alt.Chart(pdf).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#3A3A3A")
                      .encode(x=alt.X("ms:Q", title="milliseconds"), y=alt.Y("Stage:N", sort=None, title=None))
                      .properties(height=34 * len(pdf) + 20)
                      .configure_view(strokeWidth=0)
                      .configure_axis(gridColor="#EEEEEE", domainColor="#BDBDBD", labelColor="#333333", labelFont="Poppins",
                                      titleFont="Poppins", titleColor="#6B6B6B"))
        show_chart(perf_chart)
        st.caption(f"Total {r['seconds']} s · model: {r.get('llm_model', 'offline')}"
                   + (f" · {r['llm_note']}" if r.get("llm_note") else ""))
    with st.expander("Technical details (JSON)"):
        st.json({k: v for k, v in r.items() if k not in ("rewrite", "rules")})


# ---------------------------------------------------------------- sidebar
engine = get_engine()
s = engine.settings
with st.sidebar:
    st.markdown("### Settings")
    provider = st.selectbox("Language model", ["Gemini", "Groq", "Ollama (local)", "Auto (.env)", "Offline (no LLM)"])
    pmap = {"Auto (.env)": None, "Groq": "groq", "Gemini": "gemini", "Ollama (local)": "ollama",
            "Offline (no LLM)": "offline"}
    p = pmap[provider]
    key, model = None, None
    if p in ("groq", "gemini"):
        key = st.text_input(f"{provider} API key", type="password",
                            value=s.groq_api_key if p == "groq" else s.gemini_api_key) or None
    if p in ("groq", "gemini", "ollama"):
        default = {"groq": s.groq_model, "gemini": s.gemini_model, "ollama": s.ollama_model}[p]
        options = list(dict.fromkeys([default] + MODEL_CHOICES[p])) + ["Custom..."]
        model = st.selectbox("Model", options, help="If a model is retired, CivilLens switches to the next one automatically.")
        if model == "Custom...":
            model = st.text_input("Custom model name", value=default)
    llm = engine.set_llm(provider=p, api_key=key, model=model)
    if llm.available:
        c1, c2 = st.columns(2)
        if c1.button("Test connection"):
            with st.spinner("Contacting the model..."):
                ok, msg = llm.test_connection()
            st.session_state["conn"] = (ok, msg, llm.model)
        if p in ("groq", "gemini") and key and c2.button("Save key"):
            env = ROOT / ".env"
            lines = [l for l in (env.read_text().splitlines() if env.exists() else [])
                     if not l.split("=")[0].strip() in ("LLM_PROVIDER", f"{p.upper()}_API_KEY", f"{p.upper()}_MODEL")]
            lines += [f"LLM_PROVIDER={p}", f"{p.upper()}_API_KEY={key}", f"{p.upper()}_MODEL={llm.model}"]
            env.write_text("\n".join(lines) + "\n")
            st.success("Saved to .env on this computer. Next time choose 'Auto (.env)'.")
        if "conn" in st.session_state:
            ok, msg, m = st.session_state["conn"]
            (st.success if ok else st.error)(msg)
    reveal = st.checkbox("Show unmasked original (moderators only)", value=False)
    st.markdown("---")
    st.markdown("<div class='cl-label'>Active components</div>", unsafe_allow_html=True)
    for k, v in engine.components.items():
        st.markdown(f"<div style='font-size:13px;margin-bottom:6px'><b>{esc(k)}</b><br>"
                    f"<span style='color:#6B6B6B'>{esc(v)}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:#6B6B6B'>LLM cache: {CACHE_STATS['hits']} hits · "
                f"{CACHE_STATS['misses']} calls</div>", unsafe_allow_html=True)

st.markdown("<div class='cl-hero'><h1>CivilLens</h1>"
            "<p>Remove the abuse, keep the point. A Generative AI reviewer for rewrites of toxic public comments.</p></div>",
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Rewrite a comment", "Batch review (CSV)"])

# ---------------------------------------------------------------- module 1
EXAMPLES = [
    "Choose an example",
    "These useless officials are idiots, potholes not fixed for 2 years.",
    "Wow, great job as always, water department. The pipeline has been leaking for six months. Thanks for nothing.",
    "Get lost, bank staff. My loan application has been pending for 3 months. You will regret this.",
    "What the hell? Why has the college administration still not released the exam results?",
    "As a Muslim woman, I feel unsafe walking home because the street lights are broken.",
]
with tab1:
    st.markdown("#### Module 1 · Rewrite a single comment")
    st.markdown("<span style='color:#6B6B6B'>Paste a toxic comment. CivilLens rewrites it politely, then checks that "
                "the toxicity is gone and the meaning is kept.</span>", unsafe_allow_html=True)
    ex = st.selectbox("Example", EXAMPLES, label_visibility="collapsed")
    text = st.text_area("Comment", value="" if ex == EXAMPLES[0] else ex, height=110, key=f"comment_{ex}",
                        placeholder="Type or paste a public comment here...")
    with st.expander("Review your own rewrite instead (optional)"):
        own = st.text_area("Your rewrite", height=80, placeholder="Leave empty to let CivilLens write the rewrite")
    compare = st.checkbox("Compare LLM with offline mode", value=False, disabled=not llm.available,
                          help="Runs the same comment twice and shows both results side by side.")
    if st.button("Rewrite and review", key="go1"):
        if not text.strip():
            st.warning("Please enter a comment first.")
        else:
            with st.spinner("Masking, rewriting and reviewing..."):
                if own.strip():
                    rep = engine.review(text, own.strip(), use_llm=llm.available)
                    st.session_state.pop("compare", None)
                elif compare and llm.available:
                    both = engine.compare(text)
                    rep = both["llm"]
                    st.session_state["compare"] = both
                else:
                    rep = engine.rewrite_and_review(text, use_llm=llm.available)
                    st.session_state.pop("compare", None)
                st.session_state["single"] = (text, rep)
            if llm.available and llm.last_error:
                st.info(f"The LLM could not be reached, so the offline models were used. ({llm.last_error[:120]})")
    if "single" in st.session_state:
        orig, rep = st.session_state["single"]
        st.markdown("---")
        if "compare" in st.session_state:
            st.markdown("#### LLM vs offline")
            both = st.session_state["compare"]
            show_df(pd.DataFrame([{"Mode": name, "Rewrite": x["rewrite_display"], "Toxicity after": pct(x["tox_after"]),
                                   "Meaning kept": pct(x["mps"]), "CFI": round(x["cfi"], 2), "Verdict": x["verdict"],
                                   "Time (s)": x["seconds"]}
                                  for name, x in (("LLM", both["llm"]), ("Offline", both["offline"]))]))
            st.write("")
        render_report(rep, reveal_original=orig if reveal else None)

# ---------------------------------------------------------------- module 2
with tab2:
    st.markdown("#### Module 2 · Batch review from a CSV file")
    st.markdown("<span style='color:#6B6B6B'>Upload a CSV with a column of comments (<code>original_comment</code>) and, "
                "optionally, candidate rewrites (<code>candidate_rewrite</code>). Rows without a rewrite are rewritten "
                "automatically. Every row goes through the same pipeline as Module 1.</span>", unsafe_allow_html=True)
    sample_path = DATA_DIR / "input_comments_100.csv"
    a, b, c = st.columns([2, 1, 1])
    up = a.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if b.button("Use sample (100 comments)"):
        st.session_state["batch_src"] = ("sample file", pd.read_csv(sample_path))
    c.download_button("Download sample CSV", data=sample_path.read_bytes(), file_name="input_comments_100.csv",
                      mime="text/csv")
    if up is not None:
        st.session_state["batch_src"] = (up.name, pd.read_csv(up))
    if "batch_src" in st.session_state:
        name, df = st.session_state["batch_src"]
        st.caption(f"Loaded {name}: {len(df)} rows · columns: {', '.join(map(str, df.columns))}")
        cols = list(df.columns)
        o1, o2, o3, o4 = st.columns(4)
        text_col = o1.selectbox("Comment column", cols,
                                index=cols.index("original_comment") if "original_comment" in cols else 0)
        rcols = ["(none)"] + cols
        rewrite_col = o2.selectbox("Rewrite column", rcols,
                                   index=rcols.index("candidate_rewrite") if "candidate_rewrite" in cols else 0)
        auto = o3.checkbox("Auto-rewrite empty rows", value=True)
        batch_llm = o4.checkbox("Use LLM for every row", value=False, disabled=not llm.available,
                                help="Slower and uses API quota. Off = fast local models.")
        if st.button("Run batch review", key="go2"):
            bar = st.progress(0.0, text="Reviewing...")
            res = engine.review_batch(df, text_col=text_col,
                                      rewrite_col=rewrite_col if rewrite_col != "(none)" else "__no_rewrite__",
                                      auto_rewrite=auto, use_llm=batch_llm and llm.available,
                                      progress=lambda i, n: bar.progress(i / n, text=f"Reviewed {i} of {n}"))
            bar.empty()
            st.session_state["batch_res"] = (df, res, text_col)
            st.session_state["batch_perf"] = dict(res.attrs.get("perf", {}))

    if "batch_res" in st.session_state:
        src, res, text_col = st.session_state["batch_res"]
        st.markdown("---")
        vc = res.verdict.value_counts()
        m = st.columns(6)
        metric(m[0], "Comments reviewed", len(res))
        metric(m[1], "Avg toxicity before", pct(res.toxicity_before.mean()))
        metric(m[2], "Avg toxicity after", pct(res.toxicity_after.mean()))
        metric(m[3], "Avg meaning preserved", pct(res.meaning_preserved.mean()))
        metric(m[4], "Accept / Revise / Reject", f"{vc.get('Accept', 0)} / {vc.get('Revise', 0)} / {vc.get('Reject', 0)}")
        if "expected_verdict" in res.columns and res.expected_verdict.fillna("").ne("").any():
            lab = res[res.expected_verdict.fillna("") != ""]
            metric(m[5], "Agreement with labels", pct((lab.expected_verdict == lab.verdict).mean()))
        else:
            metric(m[5], "Auto-rewritten rows", int(res.rewrite_source.str.startswith("auto").sum()))
        perf = st.session_state.get("batch_perf", {})
        if perf:
            st.caption(f"Performance: {perf['rows']} comments in {perf['total_seconds']} s · "
                       f"{perf['ms_per_comment']} ms per comment · {perf['comments_per_second']} comments/second · "
                       f"{perf['auto_rewritten']} auto-rewritten")
        st.write("")
        g1, g2 = st.columns(2)
        with g1:
            v = pd.DataFrame({"Verdict": ["Accept", "Revise", "Reject"],
                              "Share": [vc.get(k, 0) / len(res) for k in ("Accept", "Revise", "Reject")]})
            show_chart(gray_bars(v, "Verdict:N", "Share:Q", title="Share of verdicts"))
        with g2:
            by = res.groupby("top_category_before")[["toxicity_before", "toxicity_after"]].mean().reset_index()
            long = pd.concat([by.assign(When="Before", Score=by.toxicity_before),
                              by.assign(When="After", Score=by.toxicity_after)])
            long["Category"] = long.top_category_before.str.capitalize()
            show_chart(gray_bars(long, "Category:N", "Score:Q", "When:N", "Average toxicity by type of original comment"))
        view = res.copy()
        for col in ("toxicity_before", "toxicity_after", "meaning_preserved"):
            view[col] = (view[col] * 100).round(0).astype(int).astype(str) + "%"
        view["cfi"] = view["cfi"].round(2)
        show_cols = [c for c in ("comment_id", "original_masked", "rewrite", "toxicity_before", "toxicity_after",
                                 "meaning_preserved", "cfi", "verdict", "expected_verdict", "flags", "rules")
                     if c in view.columns]
        f1, f2 = st.columns([1, 3])
        pick = f1.selectbox("Filter by verdict", ["All", "Accept", "Revise", "Reject"])
        shown = view if pick == "All" else view[view.verdict == pick]
        f2.caption(f"Showing {len(shown)} of {len(view)} rows. Toxic words are masked.")
        show_df(shown[show_cols], height=420)
        st.download_button("Download results CSV", data=res.drop(columns=["rewrite_raw"]).to_csv(index=False).encode("utf-8"),
                           file_name="civillens_results.csv", mime="text/csv")
        st.markdown("#### Inspect one comment")
        ids = res["comment_id"].tolist() if "comment_id" in res.columns else list(range(len(res)))
        sel = st.selectbox("Comment", ids)
        i = ids.index(sel)
        orig = str(src[text_col].iloc[i])
        rep = engine.review(orig, str(res.rewrite_raw.iloc[i]), use_llm=False)
        render_report(rep, reveal_original=orig if reveal else None)

st.markdown("<div style='color:#9A9A9A;font-size:12px;margin-top:40px'>CivilLens · Generative AI Hackathon · "
            "Alliance School of Advanced Computing · Batch CS_12</div>", unsafe_allow_html=True)
