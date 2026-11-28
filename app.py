"""
Flask application entry point.
Loads environment variables via python-dotenv and precomputed embeddings once at startup,
then serves template and API routes using the configured AI_PROVIDER (gemini or ollama).
"""

import logging
import os
import uuid
from collections import OrderedDict

from dotenv import load_dotenv

# Load .env before importing config so environment overrides take effect immediately
load_dotenv()

from flask import Flask, jsonify, request, session, render_template

import config
from rag.pipeline import (
    load_embeddings,
    answer_question,
    get_all_videos,
    get_df,
    check_ai_health,
)
from services.quiz_service import generate_quiz, score_quiz

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

ALLOWED_MODES = {"default", "simple", "step_by_step", "summary", "example"}
MAX_QUERY_LENGTH = 2000
MAX_TOPIC_LENGTH = 200

SERVICE_UNAVAILABLE_ERRORS = {
    "embeddings_not_loaded",
    "embedding_failed",
    "embedding_dim_mismatch",
    "local_embedder_missing",
    "local_embedder_failed",
    "retrieval_failed",
    "llm_failed",
    "gemini_not_configured",
    "gemini_auth_error",
    "gemini_model_error",
    "gemini_bad_request",
    "gemini_unavailable",
    "ai_timeout",
}

# Bounded in-memory store for active quizzes so correct answers stay server-side
_QUIZ_STORE: OrderedDict[str, list[dict]] = OrderedDict()
MAX_STORED_QUIZZES = 200


# Load embeddings once at startup
with app.app_context():
    ok = load_embeddings()
    if not ok:
        logger.warning(
            "Embeddings file (%s) failed to load for EMBEDDING_PROVIDER=%s, LLM_PROVIDER=%s. "
            "Questions will return an error until embeddings are available.",
            config.get_embeddings_path(),
            config.get_embedding_provider(),
            config.get_llm_provider(),
        )


def _error_http_status(error_code: str | None) -> int:
    if error_code == "rate_limited":
        return 429
    if error_code in SERVICE_UNAVAILABLE_ERRORS:
        return 503
    return 200


def _render_landing(status_code: int = 200):
    df = get_df()
    videos = get_all_videos()
    embed_model, llm_model = config.get_active_model_names()
    return (
        render_template(
            "index.html",
            course_name=config.COURSE_NAME,
            videos_count=len(videos),
            total_chunks=len(df) if df is not None else 0,
            embed_model=embed_model,
            llm_model=llm_model,
            top_k=config.TOP_K,
            ai_provider=config.get_llm_provider(),
            embedding_provider=config.get_embedding_provider(),
            llm_provider=config.get_llm_provider(),
        ),
        status_code,
    )


# ══════════════════════════════════════════════════════════════════════════
# Template routes
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def landing():
    return _render_landing(200)


@app.route("/dashboard")
def dashboard():
    df = get_df()
    videos = get_all_videos()
    embed_model, llm_model = config.get_active_model_names()
    return render_template(
        "dashboard.html",
        course_name=config.COURSE_NAME,
        videos=videos,
        total_chunks=len(df) if df is not None else 0,
        embed_model=embed_model,
        llm_model=llm_model,
        top_k=config.TOP_K,
        ai_provider=config.get_llm_provider(),
        embedding_provider=config.get_embedding_provider(),
        llm_provider=config.get_llm_provider(),
    )


# ══════════════════════════════════════════════════════════════════════════
# API: Health
# ══════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Quick liveness check for the application, embeddings, and active AI provider."""
    df = get_df()
    ai_status = check_ai_health()
    return jsonify(
        {
            "status": "ok",
            "embeddings_loaded": df is not None,
            "total_chunks": len(df) if df is not None else 0,
            "videos": int(df["number"].nunique()) if df is not None else 0,
            "course": config.COURSE_NAME,
            **ai_status,
        }
    )


# ══════════════════════════════════════════════════════════════════════════
# API: Chat / Q&A
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    POST /api/ask
    Body: { "query": str, "mode": str (optional) }
    Returns: { answer, sources, query, mode, error }
    """
    data = request.get_json(silent=True) or {}
    raw_query = data.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
    mode = data.get("mode", "default")
    if mode not in ALLOWED_MODES:
        mode = "default"

    if not query:
        return jsonify({"error": "empty_query", "answer": "Please enter a question.", "sources": []}), 400

    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            "error": "query_too_long",
            "answer": f"Please keep your question under {MAX_QUERY_LENGTH} characters.",
            "sources": [],
        }), 400

    history = session.get("history", [])
    result = answer_question(query, mode=mode, history=history)

    err = result.get("error")
    http_status = _error_http_status(err)
    if http_status != 200:
        return jsonify(result), http_status

    if not err or err == "no_relevant_content":
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["answer"]})
        session["history"] = history[-10:]
        session.modified = True

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return the current session's chat history."""
    return jsonify({"history": session.get("history", [])})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clear the session's chat history."""
    session.pop("history", None)
    session.modified = True
    return jsonify({"status": "cleared"})


# ══════════════════════════════════════════════════════════════════════════
# API: Quiz
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/quiz/generate", methods=["POST"])
def api_generate_quiz():
    """
    POST /api/quiz/generate
    Body: { "topic": str, "num_questions": int (optional, 1-10) }
    Returns: { quiz_id, topic, questions (without answers), sources, error }
    """
    data = request.get_json(silent=True) or {}
    raw_topic = data.get("topic")
    topic = raw_topic.strip() if isinstance(raw_topic, str) else ""

    if not topic:
        return jsonify({"error": "empty_topic", "questions": []}), 400

    if len(topic) > MAX_TOPIC_LENGTH:
        return jsonify({
            "error": "topic_too_long",
            "error_message": f"Please keep the topic under {MAX_TOPIC_LENGTH} characters.",
            "questions": [],
        }), 400

    try:
        num_q = int(data.get("num_questions", 5))
    except (TypeError, ValueError):
        return jsonify({
            "error": "invalid_num_questions",
            "error_message": "num_questions must be an integer between 1 and 10.",
            "questions": [],
        }), 400

    result = generate_quiz(topic, num_questions=num_q)

    err = result.get("error")
    if err and err != "no_relevant_content":
        http_status = 429 if err == "rate_limited" else 503
        return jsonify(result), http_status

    if result.get("questions"):
        quiz_id = str(uuid.uuid4())
        _QUIZ_STORE[quiz_id] = result["questions"]
        if len(_QUIZ_STORE) > MAX_STORED_QUIZZES:
            _QUIZ_STORE.popitem(last=False)

        public_questions = [
            {"question": q["question"], "options": q["options"]}
            for q in result["questions"]
        ]
        result["quiz_id"] = quiz_id
        result["questions"] = public_questions

    return jsonify(result)


@app.route("/api/quiz/score", methods=["POST"])
def api_score_quiz():
    """
    POST /api/quiz/score
    Body: { "quiz_id": str, "answers": {"0": "A", ...} }
          or legacy { "questions": [...], "answers": {"0": "A", ...} }
    Returns: { score, total, percentage, results }
    """
    data = request.get_json(silent=True) or {}
    quiz_id = data.get("quiz_id")
    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        return jsonify({"error": "invalid_answers"}), 400

    if isinstance(quiz_id, str) and quiz_id in _QUIZ_STORE:
        questions = _QUIZ_STORE[quiz_id]
    else:
        questions = data.get("questions", [])

    if not isinstance(questions, list) or not questions:
        return jsonify({"error": "no_questions"}), 400

    result = score_quiz(questions, answers)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════
# API: Course info
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/videos", methods=["GET"])
def api_videos():
    """Return the list of indexed videos."""
    return jsonify({"videos": get_all_videos(), "course": config.COURSE_NAME})


# ══════════════════════════════════════════════════════════════════════════
# Error handlers
# ══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not_found"}), 404
    return _render_landing(404)


@app.errorhandler(500)
def internal_error(e):
    logger.error("Internal server error: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal_server_error"}), 500
    return _render_landing(500)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
