"""Toxicity Auditor scoring.

Backends
  local     TF-IDF (word + character n-grams) + Logistic Regression trained on data/train_toxicity.csv,
            combined with a lexicon floor so known toxic words are never missed. Works fully offline.
  detoxify  Pre-trained Detoxify model (pip install detoxify). Used automatically if installed.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import DATA_DIR, MODEL_DIR, REPORT_DIR
from .masking import find_sarcasm, find_spans

CATEGORIES = ["insult", "profanity", "harassment", "threat", "sarcasm"]
MODEL_PATH = MODEL_DIR / "toxicity_model.joblib"
LEXICON_FLOOR = {"insult": 0.85, "profanity": 0.75, "harassment": 0.85, "threat": 0.92}
SARCASM_FLOOR = 0.70


def _features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=60000)),
    ])


def train_model(csv_path=DATA_DIR / "train_toxicity.csv", save=True, verbose=True):
    """Train the local toxicity classifier and the TF-IDF space used for meaning similarity."""
    df = pd.read_csv(csv_path)
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.category, test_size=0.2, random_state=42, stratify=df.category)
    clf = Pipeline([("features", _features()),
                    ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    bin_true, bin_pred = (y_te != "none").astype(int), (pred != "none").astype(int)
    labels = ["none"] + CATEGORIES
    metrics = {
        "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
        "binary_accuracy": round(float(accuracy_score(bin_true, bin_pred)), 4),
        "binary_f1": round(float(f1_score(bin_true, bin_pred)), 4),
        "category_accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "category_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "per_class": classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_te, pred, labels=labels).tolist()},
    }
    # refit on all data for the shipped model
    clf.fit(df.text, df.category)
    sim = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(df.text)
    bundle = {"clf": clf, "sim_vectorizer": sim, "classes": list(clf.classes_)}
    if save:
        MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        (REPORT_DIR / "toxicity_metrics.json").write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"binary accuracy {metrics['binary_accuracy']}, binary F1 {metrics['binary_f1']}, "
              f"category accuracy {metrics['category_accuracy']}, macro F1 {metrics['category_macro_f1']}")
    return bundle, metrics


def load_bundle():
    """Load the saved model; (re)train automatically if it is missing or was saved by another scikit-learn version."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            bundle["clf"].predict_proba(["test"])
            return bundle
        except Exception:
            pass
    bundle, _ = train_model(verbose=False)
    return bundle


class ToxicityScorer:
    def __init__(self, settings, bundle=None):
        self.bundle = bundle or load_bundle()
        self.clf = self.bundle["clf"]
        self.classes = list(self.clf.classes_)
        self.detox = None
        self.backend = "local"
        want = settings.toxicity_backend
        if want in ("auto", "detoxify"):
            try:
                from detoxify import Detoxify
                self.detox = Detoxify("original")
                self.backend = "detoxify + lexicon"
            except Exception:
                self.backend = "local TF-IDF + LR + lexicon"
        else:
            self.backend = "local TF-IDF + LR + lexicon"

    def score_many(self, texts):
        texts = ["" if t is None else str(t) for t in texts]
        out = []
        if self.detox is not None:
            raw = self.detox.predict(texts)
            for i, t in enumerate(texts):
                cats = {"insult": float(raw["insult"][i]), "profanity": float(raw["obscene"][i]),
                        "harassment": float(max(raw["toxicity"][i] - raw["insult"][i], 0)),
                        "threat": float(raw["threat"][i]), "sarcasm": 0.0}
                out.append(self._finish(t, cats, float(raw["toxicity"][i])))
            return out
        probs = self.clf.predict_proba(texts)
        for t, p in zip(texts, probs):
            pd_ = dict(zip(self.classes, p))
            cats = {c: float(pd_.get(c, 0.0)) for c in CATEGORIES}
            out.append(self._finish(t, cats, 1.0 - float(pd_.get("none", 0.0))))
        return out

    def score(self, text):
        return self.score_many([text])[0]

    @staticmethod
    def _finish(text, cats, overall):
        if not text.strip():
            return {"overall": 0.0, "categories": {c: 0.0 for c in CATEGORIES}, "spans": [], "sarcasm_cues": []}
        spans = find_spans(text)
        for s in spans:  # lexicon floor: a known toxic word can never score low
            cats[s["category"]] = max(cats[s["category"]], LEXICON_FLOOR[s["category"]])
        cues = find_sarcasm(text)
        if cues:
            cats["sarcasm"] = max(cats["sarcasm"], SARCASM_FLOOR)
        overall = max(overall, max(cats.values()))
        if not spans and not cues:  # no lexical evidence: trust the model but damp borderline scores
            overall = overall if overall >= 0.5 else overall * 0.6
            cats = {k: (v if v >= 0.5 else v * 0.6) for k, v in cats.items()}
        return {"overall": round(float(np.clip(overall, 0, 1)), 4),
                "categories": {k: round(float(v), 4) for k, v in cats.items()},
                "spans": spans, "sarcasm_cues": cues}
