"""
Orchestrate Router — Endpoint for Journey F.
"""
from flask import Blueprint, Response
from handlers.orchestrate_handler import orchestrate_pipeline_stream

orchestrate_bp = Blueprint("orchestrate", __name__)


@orchestrate_bp.route("/api/orchestrate/<session_id>", methods=["GET"])
def orchestrate(session_id: str):
    """
    Start the autonomous ML research loop for the given session.
    Streams progress logs via Server-Sent Events (SSE).
    """
    return Response(
        orchestrate_pipeline_stream(session_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@orchestrate_bp.route("/api/experiments/<session_id>", methods=["GET"])
def get_experiments(session_id: str):
    """
    GET /api/experiments/<session_id>
    Returns the list of experiments executed in the given session.
    """
    from memory_store import get_session
    from flask import jsonify
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    return jsonify([
        {
            "run_number": exp.run_number,
            "model_name": exp.model_name,
            "hyperparameters": exp.hyperparameters,
            "metrics": exp.metrics,
            "code_snippet": exp.code_snippet,
            "status": exp.status,
            "reason": exp.reason
        }
        for exp in session.experiments
    ]), 200

