"""
AutoML Scientist — Flask Application Entry Point.

Registers all Blueprints, configures CORS, and starts the dev server.
"""
import sys
import os

# Ensure the backend directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS

from routers.upload_router import upload_bp
from routers.codegen_router import codegen_bp
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)

    # ── CORS (allow React frontend) ───────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Register Blueprints ───────────────────────────────────────────────
    from routers.upload_router import upload_bp
    from routers.codegen_router import codegen_bp
    from routers.orchestrate_router import orchestrate_bp
    app.register_blueprint(upload_bp)
    app.register_blueprint(codegen_bp)
    app.register_blueprint(orchestrate_bp)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "automl-scientist"}), 200

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "message": "Welcome to the AutoML Scientist API",
            "endpoints": {
                "upload": "POST /api/upload",
                "codegen": "POST /api/codegen/<session_id>",
                "orchestrate": "GET /api/orchestrate/<session_id>",
                "download": "GET /api/download/<session_id>",
                "health": "GET /api/health"
            }
        }), 200

    @app.route("/static/plots/<filename>", methods=["GET"])
    def serve_plot(filename: str):
        from flask import send_from_directory
        plots_dir = os.path.join(os.getcwd(), "static", "plots")
        return send_from_directory(plots_dir, filename)

    @app.route("/api/download/<session_id>", methods=["GET"])
    def download_model(session_id: str):
        """Download the best model for a given session."""
        from flask import send_file
        model_path = os.path.join("models", f"{session_id}_best_model.pkl")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found for this session."}), 404
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"automl_model_{session_id[:8]}.pkl"
        )

    # ── Error Handlers ────────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    return app


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    print(f"[START] AutoML Scientist backend running on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
