"""
In-memory session state management for AutoML Scientist.

No database — all state lives in a dict keyed by session_id.
State is lost when the server restarts (by design for hackathon).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DatasetState:
    """Holds everything we know about the uploaded dataset."""
    session_id: str
    filename: str
    df_raw: Any                           # Original pandas DataFrame
    df_clean: Any = None                  # Preprocessed DataFrame (after cleaning)
    problem_type: str = ""                # "classification" | "regression"
    target_column: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentState:
    """One experiment run (model train + evaluate)."""
    run_number: int
    model_name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""
    status: str = "Pending"               # "Accepted" | "Improved" | "Failed"
    reason: str = ""


@dataclass
class SessionState:
    """Top-level session container."""
    dataset: Optional[DatasetState] = None
    experiments: List[ExperimentState] = field(default_factory=list)
    narration_logs: List[str] = field(default_factory=list)
    generated_code: str = ""              # AI-generated clean(df) code
    report_markdown: str = ""


# ── Global Store ──────────────────────────────────────────────────────────────

_session_store: Dict[str, SessionState] = {}


# ── Helper Functions ──────────────────────────────────────────────────────────

def create_session(session_id: Optional[str] = None) -> str:
    """Create a new empty session. Returns session_id."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    _session_store[session_id] = SessionState()
    return session_id


def get_session(session_id: str) -> Optional[SessionState]:
    """Retrieve session by ID. Returns None if not found."""
    return _session_store.get(session_id)


def update_session(session_id: str, **kwargs) -> bool:
    """Update fields on an existing session. Returns False if session missing."""
    session = _session_store.get(session_id)
    if session is None:
        return False
    for key, value in kwargs.items():
        if hasattr(session, key):
            setattr(session, key, value)
    return True


def delete_session(session_id: str) -> bool:
    """Remove a session from the store."""
    return _session_store.pop(session_id, None) is not None


def list_sessions() -> List[str]:
    """Return all active session IDs."""
    return list(_session_store.keys())
