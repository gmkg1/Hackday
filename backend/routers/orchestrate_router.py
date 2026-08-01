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
