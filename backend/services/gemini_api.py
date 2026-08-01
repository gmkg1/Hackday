"""
Gemini API Service — Pure API wrapper for Google Gemini.

This service handles ONLY the API communication with Gemini.
It knows nothing about datasets, prompts, or cleaning logic.

Responsibilities:
    - Gemini client initialization
    - Sending prompts and receiving responses
    - Error handling for API failures
    - Response text extraction
"""
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL


# ── Gemini Client (lazy-initialized) ─────────────────────────────────────────

_client = None


def _get_client() -> genai.Client:
    """Lazy-init the Gemini client. Raises if API key is missing."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to backend/.env or set it as an environment variable."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def call_gemini(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Send a prompt to Gemini and return the response.

    Args:
        prompt: The full prompt string to send.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
                     Default 0.2 for code generation.
        max_output_tokens: Optional cap on response length.

    Returns:
        dict with:
            - "text": Raw response text from Gemini
            - "error": None on success, error message on failure
    """
    if not GEMINI_API_KEY:
        return {
            "text": None,
            "error": "GEMINI_API_KEY not set",
        }

    try:
        client = _get_client()

        config_kwargs = {"temperature": temperature}
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = max_output_tokens

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        return {
            "text": response.text,
            "error": None,
        }

    except Exception as e:
        return {
            "text": None,
            "error": str(e),
        }


def is_api_available() -> bool:
    """Check if the Gemini API key is configured."""
    return bool(GEMINI_API_KEY)
