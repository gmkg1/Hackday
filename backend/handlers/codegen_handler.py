"""
Code Generation Handler — Journey D (AI Generates Cleaning Code).

Retrieves session metadata and calls the LLM client to produce
a dataset-specific clean(df) function.
"""
from typing import Dict, Any, Tuple

from services.code_generator import generate_cleaning_code, regenerate_cleaning_code
from services.executor import run_generated_code
from memory_store import get_session, update_session
from config import MAX_CODE_RETRIES


def handle_codegen(session_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Generate AI cleaning code for the dataset in the given session.

    Pipeline:
      1. Retrieve session + dataset metadata
      2. Call Gemini to generate clean(df) code
      3. (Optional) Validate by executing on a small sample
      4. Store generated code in session state
      5. Return the code + explanation

    Returns:
        (response_dict, http_status_code)
    """

    # ── Step 1: Retrieve session ──────────────────────────────────────────
    session = get_session(session_id)
    if session is None:
        return {"error": f"Session '{session_id}' not found."}, 404

    if session.dataset is None:
        return {"error": "No dataset uploaded for this session."}, 400

    # ── Step 2: Get LLM metadata ─────────────────────────────────────────
    llm_metadata = session.dataset.summary.get("llm_metadata")
    if not llm_metadata:
        return {"error": "Dataset metadata not available. Re-upload the file."}, 400

    # ── Step 3: Generate code via Gemini ──────────────────────────────────
    result = generate_cleaning_code(llm_metadata)
    code = result["code"]
    error = result.get("error")

    # ── Step 4: Validate by running on a small sample ─────────────────────
    # Take a small slice to test execution without running on the full dataset
    df_raw = session.dataset.df_raw
    sample_size = min(50, len(df_raw))
    df_sample = df_raw.head(sample_size).copy()

    cleaned_df, exec_error = run_generated_code(code, df_sample)

    # ── Step 5: Self-correction retry loop ────────────────────────────────
    retries = 0
    while exec_error and retries < MAX_CODE_RETRIES:
        retries += 1
        retry_result = regenerate_cleaning_code(
            metadata=llm_metadata,
            previous_code=code,
            error_traceback=exec_error,
        )
        code = retry_result["code"]
        error = retry_result.get("error")

        # Re-test the corrected code
        cleaned_df, exec_error = run_generated_code(code, df_sample)

    # ── Step 6: Store generated code in session ───────────────────────────
    session.generated_code = code

    # ── Step 7: Build response ────────────────────────────────────────────
    response = {
        "session_id": session_id,
        "code": code,
        "explanation": result.get("explanation", ""),
        "retries_used": retries,
        "validation": {
            "tested": True,
            "passed": exec_error is None,
            "error": exec_error,
            "sample_rows_tested": sample_size,
        },
    }

    if exec_error:
        response["warning"] = (
            f"Code failed validation after {retries} retries. "
            "It may still work on the full dataset."
        )

    return response, 200
