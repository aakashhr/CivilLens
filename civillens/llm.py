"""Thin LLM client for Groq, Gemini and Ollama using plain HTTP (no SDKs needed).

Every call asks for JSON output. If the provider is unavailable, calls raise LLMError and
the pipeline falls back to its offline methods, so the app never breaks during a demo.
"""
import json
import re

import requests


class LLMError(Exception):
    pass


# Models tried in order. If one is retired or blocked (HTTP 404/400 "model"), the next one is used automatically.
MODEL_CHOICES = {
    "groq": ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "gemini": ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-3.6-flash"],
    "ollama": ["llama3.1", "qwen2.5", "mistral"],
}
_MODEL_ERR = ("404", "not found", "does not exist", "no longer available", "decommissioned", "not supported",
              "model_not_found", "is not found")


_CACHE = {}          # (provider, system, user, temperature) -> JSON reply; shared across the app session
CACHE_STATS = {"hits": 0, "misses": 0}


class LLMClient:
    def __init__(self, settings, provider=None, api_key=None, model=None):
        self.s = settings
        p = (provider or settings.llm_provider or "auto").lower()
        if p == "auto":
            if api_key or settings.groq_api_key:
                p = "groq"
            elif settings.gemini_api_key:
                p = "gemini"
            else:
                p = "offline"
        self.provider = p
        self.api_key = api_key or {"groq": settings.groq_api_key, "gemini": settings.gemini_api_key}.get(p, "")
        self.model = model or {"groq": settings.groq_model, "gemini": settings.gemini_model,
                               "ollama": settings.ollama_model}.get(p, "")
        self.last_error = ""
        self.note = ""          # e.g. "switched from X to Y"
        self.last_seconds = 0.0

    @property
    def available(self):
        if self.provider in ("groq", "gemini"):
            return bool(self.api_key)
        return self.provider == "ollama"

    @property
    def label(self):
        return f"{self.provider} ({self.model})" if self.available else "offline (no LLM)"

    # ------------------------------------------------------------------
    def chat_json(self, system, user, temperature=0.2, timeout=40):
        """Call the model; on a retired/blocked model, fall back to the next model in MODEL_CHOICES."""
        import time
        if not self.available:
            raise LLMError("No LLM configured")
        key = (self.provider, system, user, temperature)
        if key in _CACHE:
            CACHE_STATS["hits"] += 1
            self.last_seconds = 0.0
            return _CACHE[key]
        CACHE_STATS["misses"] += 1
        tried = []
        candidates = [self.model] + [m for m in MODEL_CHOICES.get(self.provider, []) if m != self.model]
        for model in candidates:
            self.model = model
            t0 = time.time()
            try:
                out = self._call(system, user, temperature, timeout)
                _CACHE[key] = out
                self.last_seconds = round(time.time() - t0, 2)
                self.last_error = ""
                if tried:
                    self.note = f"Model {tried[0]} unavailable; switched to {model}"
                return out
            except LLMError as e:
                msg = str(e).lower()
                tried.append(model)
                if not any(k in msg for k in _MODEL_ERR):
                    raise
        raise LLMError(self.last_error or "No working model found")

    def test_connection(self):
        """Returns (ok, message)."""
        import time
        try:
            r = self.chat_json("Reply with JSON only.", f'Return {{"status": "ok", "check": {time.time():.0f}}}', timeout=20)
            return True, f"Connected to {self.provider} · {self.model} ({self.last_seconds} s)" + (
                f" — {self.note}" if self.note else "")
        except LLMError as e:
            return False, str(e)[:300]

    def _call(self, system, user, temperature, timeout):
        try:
            if self.provider == "groq":
                text = self._groq(system, user, temperature, timeout)
            elif self.provider == "gemini":
                text = self._gemini(system, user, temperature, timeout)
            else:
                text = self._ollama(system, user, temperature, timeout)
            return parse_json(text)
        except LLMError as e:
            self.last_error = str(e)
            raise
        except Exception as e:  # network, quota, bad JSON
            self.last_error = f"{type(e).__name__}: {e}"
            raise LLMError(self.last_error) from e

    def _groq(self, system, user, temperature, timeout):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": temperature, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=timeout)
        if r.status_code != 200:
            raise LLMError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

    def _gemini(self, system, user, temperature, timeout):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        r = requests.post(
            url, params={"key": self.api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}},
            timeout=timeout)
        if r.status_code != 200:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _ollama(self, system, user, temperature, timeout):
        r = requests.post(
            f"{self.s.ollama_url}/api/chat",
            json={"model": self.model, "stream": False, "format": "json", "options": {"temperature": temperature},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=timeout * 3)
        if r.status_code != 200:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"]


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise LLMError("LLM did not return JSON")
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# Prompts. The LLM only ever sees the MASKED original (toxic words replaced by tags).
REWRITE_SYSTEM = """You are a civil-language editor for a public comment platform.
Rewrite the user's comment so that it is polite and free of insults, profanity, threats, harassment and sarcasm,
while keeping EXACTLY the same complaint, stance (for/against), target (who or what), facts, numbers and time periods.
Toxic words in the input are masked as tags like [INSULT], [PROFANITY], [HARASSMENT], [THREAT]; never output these tags.
Do not add any new facts, names, numbers or accusations. Do not turn a complaint into praise.
Return JSON: {"rewrite": "<the rewritten comment>", "removed": ["<short description of each toxic element removed>"]}"""

ANALYSIS_SYSTEM = """You review whether a rewrite of a toxic public comment is safe and faithful.
The ORIGINAL is masked (toxic words appear as tags like [INSULT]). Compare it with the REWRITE.
Return JSON with these keys:
{"original": {"claim": str, "stance": "against"|"for"|"neutral", "target": str, "grievance": str},
 "rewrite":  {"claim": str, "stance": "against"|"for"|"neutral", "target": str, "grievance": str},
 "same_claim": bool, "same_stance": bool, "same_target": bool, "same_grievance": bool,
 "new_facts": [str], "sarcasm_or_mockery": bool, "hidden_toxicity": bool,
 "intent_recovered": bool, "notes": str}
"intent_recovered" = true if a reader of ONLY the rewrite would understand what the original writer wanted.
Never quote abusive words; refer to them by category."""

EXPLAIN_SYSTEM = """You are the judge in a comment-moderation system. Using the scores, flags and community rules given,
write a short plain-English explanation (2-3 sentences) of why the rewrite got its verdict.
Never repeat abusive words. Return JSON: {"explanation": "<text>"}"""
