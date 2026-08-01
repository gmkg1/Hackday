"""
Configuration for AutoML Scientist backend.
All settings loaded from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ── LLM Provider ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Upload Limits ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Self-Correction ──────────────────────────────────────────────────────────
MAX_CODE_RETRIES = 2

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
