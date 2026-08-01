"""
Upload Router — Journey A (Upload Dataset).

Blueprint that registers POST /api/upload.
Thin layer: parses the request, delegates to upload_handler.
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from handlers.upload_handler import handle_upload


upload_bp = Blueprint("upload", __name__, url_prefix="/api")


@upload_bp.route("/upload", methods=["POST"])
def upload_dataset():
    """
    POST /api/upload

    Accepts multipart/form-data with:
        - file: CSV file (required)
        - session_id: Optional client-provided session UUID
        - target_column: Optional user-specified target column name

    Returns JSON with session_id, profile summary, and problem type.
    """
    # ── Parse request ─────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Include a 'file' field."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    session_id = request.form.get("session_id", None)
    target_column = request.form.get("target_column", None)

    # ── Delegate to handler ───────────────────────────────────────────────
    response, status_code = handle_upload(
        file_storage=file,
        filename=filename,
        session_id=session_id,
        target_column=target_column,
    )

    return jsonify(response), status_code
