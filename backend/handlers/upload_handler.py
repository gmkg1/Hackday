"""
Upload Handler — Journey A (Upload) + B (Profile) + C (Problem Type Detection).

Orchestrates validation, profiling, and problem detection.
No direct Flask imports — receives parsed data from the router.
"""
import uuid
import pandas as pd
from io import StringIO
from typing import Dict, Any, Tuple

from services.profiler import profile_dataset, detect_problem_type, build_llm_metadata
from memory_store import (
    create_session,
    get_session,
    DatasetState,
    SessionState,
)
import memory_store


# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".csv"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def handle_upload(
    file_storage,
    filename: str,
    session_id: str = None,
    target_column: str = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Full upload pipeline:
      1. Validate the file (extension, size)
      2. Parse CSV into DataFrame
      3. Profile the dataset (Journey B)
      4. Detect problem type (Journey C)
      5. Store everything in memory
      6. Return JSON-serializable response

    Args:
        file_storage: Werkzeug FileStorage object from request.files
        filename: Original filename
        session_id: Optional client-provided session ID
        target_column: Optional user-specified target column

    Returns:
        (response_dict, http_status_code)
    """

    # ── Step 1: Validate ──────────────────────────────────────────────────
    error = _validate_file(filename, file_storage)
    if error:
        return {"error": error}, 400

    # ── Step 2: Parse CSV ─────────────────────────────────────────────────
    try:
        # Read the file content; handle encoding gracefully
        raw_content = file_storage.read()
        file_size = len(raw_content)

        if file_size > MAX_FILE_SIZE_BYTES:
            return {"error": f"File exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit."}, 400

        # Try UTF-8 first, fall back to latin-1
        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_content.decode("latin-1")

        df = pd.read_csv(StringIO(text))

    except Exception as e:
        return {"error": f"Failed to parse CSV: {str(e)}"}, 400

    if df.empty:
        return {"error": "Uploaded CSV is empty."}, 400

    # ── Step 3: Profile dataset (Journey B) ───────────────────────────────
    profile = profile_dataset(df)

    # ── Step 4: Detect problem type (Journey C) ───────────────────────────
    problem_info = detect_problem_type(df, target_column=target_column)

    # ── Step 5: Build LLM metadata (for Journey D later) ─────────────────
    llm_metadata = build_llm_metadata(df, profile, problem_info)

    # ── Step 6: Store in memory ───────────────────────────────────────────
    if not session_id:
        session_id = str(uuid.uuid4())

    create_session(session_id)
    session = get_session(session_id)

    session.dataset = DatasetState(
        session_id=session_id,
        filename=filename,
        df_raw=df,
        problem_type=problem_info["problem_type"],
        target_column=problem_info["target_column"],
        summary=profile,
    )

    # Stash LLM metadata on the session for use by codegen handler
    session.dataset.summary["llm_metadata"] = llm_metadata

    # ── Step 7: Build response ────────────────────────────────────────────
    response = {
        "session_id": session_id,
        "filename": filename,
        "problem_type": problem_info["problem_type"],
        "target_column": problem_info["target_column"],
        "summary": {
            "num_rows": profile["num_rows"],
            "num_cols": profile["num_cols"],
            "columns": profile["columns"],
            "column_types": profile["column_types"],
            "missing_values": profile["missing_values"],
            "missing_percentages": profile["missing_percentages"],
            "duplicate_rows": profile["duplicate_rows"],
            "cardinality": profile["cardinality"],
            "basic_stats": profile["basic_stats"],
            "sample_rows": profile["sample_rows"],
        },
    }

    # Add classification-specific or regression-specific info
    if problem_info["problem_type"] == "classification":
        response["num_classes"] = problem_info.get("num_classes")
        response["class_distribution"] = problem_info.get("class_distribution")
    else:
        response["target_stats"] = problem_info.get("target_stats")

    return response, 200


# ── Private Helpers ───────────────────────────────────────────────────────────

def _validate_file(filename: str, file_storage) -> str | None:
    """
    Validate file extension. Returns error message or None if valid.
    """
    if not filename:
        return "No filename provided."

    # Check extension
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return f"Invalid file type '{ext}'. Only .csv files are accepted."

    return None
