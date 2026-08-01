"""
Groq API Service — Pure API wrapper for Groq SDK.

Responsibilities:
    - Groq client initialization
    - Sending prompts and receiving JSON responses
    - Error handling for API failures
"""
import json
from typing import Dict, Any, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL


# ── Groq Client (lazy-initialized) ───────────────────────────────────────────

_client = None


def _get_client() -> Groq:
    """Lazy-init the Groq client. Raises if API key is missing."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to backend/.env or set it as an environment variable."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def call_groq_json(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 4096,
) -> Dict[str, Any]:
    """
    Send a prompt to Groq forcing a JSON object response.

    Args:
        prompt: The full prompt string.
        temperature: Controls randomness (default 0.2 for analytical tasks).
        max_tokens: Maximum output tokens.

    Returns:
        dict with:
            - "json": parsed JSON dictionary from the response, or None on failure
            - "error": None on success, error message on failure
    """
    if not GROQ_API_KEY:
        return {
            "json": None,
            "error": "GROQ_API_KEY not set",
        }

    try:
        client = _get_client()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized JSON-only output agent. You must output ONLY valid JSON. No markdown, no explanations outside the JSON."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        content = chat_completion.choices[0].message.content
        
        try:
            parsed_json = json.loads(content)
            return {
                "json": parsed_json,
                "error": None,
            }
        except json.JSONDecodeError as e:
            return {
                "json": None,
                "error": f"Failed to parse JSON response: {e}. Raw content: {content[:200]}...",
            }

    except Exception as e:
        return {
            "json": None,
            "error": str(e),
        }


def is_groq_available() -> bool:
    """Check if the Groq API key is configured."""
    return bool(GROQ_API_KEY)
