"""
Groq API Service — Pure API wrapper for Groq SDK.

Responsibility:
    - Groq client initialization
    - Sending prompts and receiving JSON responses
    - Error handling for API failures
"""
import json
import logging
from typing import Dict, Any, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("app")

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

        msg = chat_completion.choices[0].message
        content = msg.content
        
        # Fallback to reasoning/reasoning_content attributes for reasoning-capable models (e.g. DeepSeek-R1)
        if not content:
            content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if not content and hasattr(msg, "model_extra"):
                content = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")

        if not content:
            return {
                "json": None,
                "error": f"Groq API returned empty or null content choice. Message object: {msg}"
            }

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


def check_groq_api_status() -> Dict[str, Any]:
    """
    Test the Groq API key by making a lightweight test request.
    Returns a dict indicating success, error message, and details.
    """
    if not GROQ_API_KEY:
        return {"status": "inactive", "message": "GROQ_API_KEY is missing."}
        
    try:
        client = _get_client()
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            model=GROQ_MODEL,
            max_tokens=10,
        )
        msg = chat_completion.choices[0].message
        content = msg.content
        
        # Fallback to reasoning content attributes for reasoning models
        if not content:
            content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if not content and hasattr(msg, "model_extra"):
                content = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
                
        if content:
            return {"status": "working", "message": "Groq API validated successfully."}
        return {"status": "error", "message": "Groq API returned empty response."}
    except Exception as e:
        return {"status": "error", "message": f"Groq API connection failed: {str(e)}"}
