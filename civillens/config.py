"""Settings, read from environment variables or a .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

try:  # optional
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


@dataclass
class Settings:
    # LLM provider: auto | groq | gemini | ollama | offline
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    # Toxicity backend: auto | detoxify | local
    toxicity_backend: str = os.getenv("TOXICITY_BACKEND", "auto")
    # Meaning embeddings: auto | sbert | tfidf
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")
    sbert_model: str = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Verdict thresholds
    accept_threshold: float = float(os.getenv("CFI_ACCEPT", "0.75"))
    reject_threshold: float = float(os.getenv("CFI_REJECT", "0.50"))
    residual_toxicity: float = float(os.getenv("RESIDUAL_TOXICITY", "0.50"))
