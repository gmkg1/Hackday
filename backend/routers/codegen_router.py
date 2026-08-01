"""
Code Generation Router — Journey D (AI Generates Cleaning Code).

Blueprint that registers POST /api/codegen/<session_id>.
Thin layer: extracts session_id from URL, delegates to codegen_handler.
"""
from flask import Blueprint, jsonify

from handlers.codegen_handler import handle_codegen


codegen_bp = Blueprint("codegen", __name__, url_prefix="/api")


@codegen_bp.route("/codegen/<session_id>", methods=["POST"])
def generate_code(session_id: str):
    """
    POST /api/codegen/<session_id>

    Triggers AI code generation for the dataset in the given session.
    The Gemini model receives rich metadata (stats, correlations, diverse
    sample rows) to write a feasible clean(df) function.

    Returns JSON with:
        - code: the generated Python code
        - explanation: brief description of what the code does
        - retries_used: number of self-correction retries
        - validation: whether the code passed sample execution
    """
    response, status_code = handle_codegen(session_id)
    return jsonify(response), status_code
