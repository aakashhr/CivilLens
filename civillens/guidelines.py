"""Guidelines RAG: retrieves the community rules that apply to a verdict.

Rules are embedded once (FAISS + Sentence-BERT if installed, otherwise a TF-IDF index) and the
judge queries them with a description of the problems found.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DATA_DIR

# Which rule each red-team flag maps to (used together with retrieval for precise citations)
FLAG_RULES = {
    "residual_toxicity": "G1", "profanity": "G2", "threat": "G3", "harassment": "G4", "sarcasm": "G5",
    "over_cleaned": "G6", "stance_flipped": "G7", "invented_facts": "G8", "identity": "G9",
    "fluency": "G10",
}


def load_rules(path=DATA_DIR / "guidelines.md"):
    text = path.read_text(encoding="utf-8")
    rules = []
    for m in re.finditer(r"^## (G\d+) (.+?)\n(.+?)(?=\n## |\Z)", text, flags=re.S | re.M):
        rules.append({"id": m.group(1), "title": m.group(2).strip(), "text": " ".join(m.group(3).split())})
    return rules


class GuidelineIndex:
    def __init__(self, embedder=None):
        self.rules = load_rules()
        self.by_id = {r["id"]: r for r in self.rules}
        docs = [f"{r['title']}. {r['text']}" for r in self.rules]
        self.backend = "TF-IDF index"
        self.faiss_index = None
        self.sbert = getattr(embedder, "sbert", None)
        if self.sbert is not None:
            try:
                import faiss
                vecs = self.sbert.encode(docs, normalize_embeddings=True).astype("float32")
                self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])
                self.faiss_index.add(vecs)
                self.backend = "FAISS + Sentence-BERT"
            except Exception:
                self.faiss_index = None
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
        self.mat = self.vec.transform(docs)

    def retrieve(self, query, k=2):
        if not query.strip():
            return []
        if self.faiss_index is not None:
            q = self.sbert.encode([query], normalize_embeddings=True).astype("float32")
            _, idx = self.faiss_index.search(q, k)
            return [self.rules[i] for i in idx[0]]
        sims = (self.mat @ self.vec.transform([query]).T).toarray().ravel()
        return [self.rules[i] for i in np.argsort(-sims)[:k] if sims[i] > 0]

    def rules_for(self, flags, query):
        ids = [FLAG_RULES[f] for f in flags if f in FLAG_RULES]
        out, seen = [], set()
        for rid in ids:
            if rid not in seen:
                seen.add(rid); out.append(self.by_id[rid])
        for r in self.retrieve(query, k=2):
            if r["id"] not in seen and len(out) < 4:
                seen.add(r["id"]); out.append(r)
        return out
