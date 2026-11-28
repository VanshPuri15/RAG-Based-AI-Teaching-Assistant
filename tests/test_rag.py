"""
Unit tests for the RAG pipeline.
Run with: python -m pytest tests/ -v
Requires: embeddings.joblib present, Ollama NOT required (mocked).
"""

import sys
import os
import json
import types
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

# ── Test Environment Isolation ────────────────────────────────────────────

_ISOLATED_ENV_VARS = (
    "AI_PROVIDER",
    "EMBEDDING_PROVIDER",
    "LLM_PROVIDER",
    "EMBEDDINGS_PATH",
    "GEMINI_API_KEY",
    "GEMINI_GENERATION_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "GEMINI_EMBEDDING_DIM",
    "GEMINI_TIMEOUT",
    "GEMINI_MAX_RETRIES",
    "RELEVANCE_THRESHOLD",
    "TOP_K",
    "LOCAL_EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_DIM",
    "LOCAL_EMBEDDING_THREADS",
    "LOCAL_EMBEDDING_BATCH_SIZE",
    "LOCAL_EMBEDDING_CACHE_DIR",
    "OLLAMA_BASE_URL",
    "EMBED_MODEL",
    "LLM_MODEL",
    "OLLAMA_TIMEOUT",
    "HOST",
    "PORT",
    "SECRET_KEY",
    "FLASK_DEBUG",
    "COURSE_NAME",
)


@pytest.fixture(autouse=True)
def isolate_provider_env(monkeypatch):
    """
    Isolate every test from `.env` values loaded at import time so unit tests
    never inherit local provider choices or a live GEMINI_API_KEY, and restore
    all environment/config/pipeline state cleanly after each test.
    """
    import config
    from rag import gemini_client, pipeline

    for var in _ISOLATED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(config, "AI_PROVIDER", "ollama")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr(config, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setattr(config, "GEMINI_EMBEDDING_DIM", 768)
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_DIM", 384)
    monkeypatch.setattr(config, "EMBED_MODEL", "bge-m3")
    monkeypatch.setattr(config, "LLM_MODEL", "llama3.2")
    monkeypatch.setattr(config, "TOP_K", 5)

    gemini_client._client_cache.clear()

    def _forbid_live_gemini_client(*args, **kwargs):
        raise AssertionError(
            "Unit tests must mock rag.gemini_client.genai.Client; live Gemini API calls are forbidden."
        )

    monkeypatch.setattr(gemini_client.genai, "Client", _forbid_live_gemini_client)
    prev_df = pipeline._df
    prev_matrix = pipeline._embedding_matrix
    prev_df_id = pipeline._embedding_matrix_df_id

    yield

    gemini_client._client_cache.clear()
    pipeline._df = prev_df
    pipeline._embedding_matrix = prev_matrix
    pipeline._embedding_matrix_df_id = prev_df_id


# ── Helpers ───────────────────────────────────────────────────────────────

def make_df(n=10, dim=16):
    """Create a small synthetic embeddings DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append({
            "chunk_id": i,
            "number": str((i % 5) + 1),
            "title": f"Test Video {(i % 5) + 1}",
            "start": float(i * 10),
            "end": float(i * 10 + 9),
            "text": f"This is test transcript chunk number {i}.",
            "embedding": rng.random(dim).tolist(),
        })
    return pd.DataFrame(rows)


# ── Retrieval tests ────────────────────────────────────────────────────────

class TestRetrieval:

    def test_find_top_k_returns_correct_count(self):
        from rag.retrieval import find_top_k
        df = make_df(20)
        # Use first row's embedding as query
        query_emb = df["embedding"].iloc[0]
        top_df, scores = find_top_k(df, query_emb, top_k=5)
        # May return fewer if threshold not met, but should not return more than top_k
        if top_df is not None:
            assert len(top_df) <= 5

    def test_find_top_k_returns_none_on_none_embedding(self):
        from rag.retrieval import find_top_k
        df = make_df(5)
        result_df, scores = find_top_k(df, None, top_k=3)
        assert result_df is None
        assert scores is None

    def test_format_sources_empty_df(self):
        from rag.retrieval import format_sources
        df = make_df(0)
        result = format_sources(df)
        assert result == []

    def test_format_sources_has_required_keys(self):
        from rag.retrieval import find_top_k, format_sources
        df = make_df(10)
        query_emb = df["embedding"].iloc[0]
        top_df, _ = find_top_k(df, query_emb, top_k=3)
        if top_df is not None and not top_df.empty:
            sources = format_sources(top_df)
            for s in sources:
                assert "video_number" in s
                assert "title" in s
                assert "start" in s
                assert "end" in s
                assert "start_fmt" in s
                assert "end_fmt" in s
                assert "excerpt" in s
                assert "text" not in s

    def test_format_sources_include_full_text(self):
        from rag.retrieval import format_sources
        df = make_df(1)
        df["text"] = ["A" * 300]
        sources = format_sources(df, include_full_text=True)
        assert len(sources) == 1
        assert len(sources[0]["text"]) == 300
        assert len(sources[0]["excerpt"]) < 230

    def test_fmt_time(self):
        from rag.retrieval import _fmt_time
        assert _fmt_time(0) == "00:00"
        assert _fmt_time(60) == "01:00"
        assert _fmt_time(90) == "01:30"
        assert _fmt_time(3661) == "61:01"

    def test_embed_query_handles_empty_string(self):
        from rag.retrieval import embed_query
        result = embed_query("")
        assert result is None

    def test_embed_query_handles_whitespace(self):
        from rag.retrieval import embed_query
        result = embed_query("   ")
        assert result is None

    @patch("rag.retrieval.requests.post")
    def test_embed_query_returns_embedding_on_success(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from rag.retrieval import embed_query
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"embeddings": [[0.1] * 1024]}
        result = embed_query("What is CSS?")
        mock_post.assert_called_once()
        assert result is not None
        assert len(result) == 1024

    @patch("rag.retrieval.requests.post")
    def test_embed_query_handles_connection_error(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        import requests as req
        from rag.retrieval import embed_query
        mock_post.side_effect = req.exceptions.ConnectionError("Connection refused")
        result = embed_query("test query")
        mock_post.assert_called_once()
        assert result is None

    @patch("rag.retrieval.requests.post")
    def test_embed_query_handles_timeout(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        import requests as req
        from rag.retrieval import embed_query
        mock_post.side_effect = req.exceptions.Timeout("timed out")
        result = embed_query("test query")
        mock_post.assert_called_once()
        assert result is None


# ── Generation tests ───────────────────────────────────────────────────────

class TestGeneration:

    def test_build_prompt_empty_sources(self):
        from rag.generation import build_prompt
        prompt = build_prompt("What is CSS?", [], mode="default")
        assert "What is CSS?" in prompt
        assert "No relevant" in prompt or "not appear" in prompt or "does not" in prompt

    def test_build_prompt_with_sources(self):
        from rag.generation import build_prompt
        sources = [{
            "video_number": "14",
            "title": "Introduction to CSS",
            "start_fmt": "01:00",
            "end_fmt": "02:00",
            "excerpt": "CSS stands for Cascading Style Sheets.",
        }]
        prompt = build_prompt("What is CSS?", sources, mode="default")
        assert "What is CSS?" in prompt
        assert "Introduction to CSS" in prompt
        assert "CSS stands for Cascading Style Sheets." in prompt

    def test_build_prompt_modes(self):
        from rag.generation import build_prompt
        sources = [{"video_number": "1", "title": "Test", "start_fmt": "00:00", "end_fmt": "01:00", "excerpt": "test"}]
        for mode in ["simple", "step_by_step", "summary", "example"]:
            prompt = build_prompt("test question", sources, mode=mode)
            assert len(prompt) > 50

    @patch("rag.generation.requests.post")
    def test_call_llm_returns_response(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from rag.generation import call_llm
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"response": "This is the answer."}
        result = call_llm("test prompt")
        mock_post.assert_called_once()
        assert result == "This is the answer."

    @patch("rag.generation.requests.post")
    def test_call_llm_handles_connection_error(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        import requests as req
        from rag.generation import call_llm
        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        result = call_llm("test prompt")
        mock_post.assert_called_once()
        assert result is None

    @patch("rag.generation.requests.post")
    def test_call_llm_handles_timeout(self, mock_post, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        import requests as req
        from rag.generation import call_llm
        mock_post.side_effect = req.exceptions.Timeout("timed out")
        result = call_llm("test prompt")
        mock_post.assert_called_once()
        assert result is None


# ── Pipeline tests ─────────────────────────────────────────────────────────

class TestPipeline:

    def test_answer_question_empty_query(self):
        from rag import pipeline
        pipeline._df = make_df()
        result = pipeline.answer_question("")
        assert result["error"] == "empty_query"

    def test_answer_question_whitespace_query(self):
        from rag import pipeline
        pipeline._df = make_df()
        result = pipeline.answer_question("   ")
        assert result["error"] == "empty_query"

    def test_answer_question_embeddings_not_loaded(self):
        from rag import pipeline
        pipeline._df = None
        result = pipeline.answer_question("What is HTML?")
        assert result["error"] == "embeddings_not_loaded"

    @patch("rag.pipeline.embed_query", return_value=None)
    def test_answer_question_embedding_fails(self, mock_embed):
        from rag import pipeline
        pipeline._df = make_df()
        result = pipeline.answer_question("What is HTML?")
        assert result["error"] == "embedding_failed"

    def test_answer_question_expands_followup_query_for_retrieval(self):
        from rag import pipeline
        df = make_df(5)
        pipeline._df = df
        history = [
            {"role": "user", "content": "What are semantic tags in HTML?"},
            {"role": "assistant", "content": "They give meaning to structure."},
        ]
        with patch("rag.pipeline.embed_query", return_value=df["embedding"].iloc[0]) as mock_embed, \
             patch("rag.pipeline.call_llm", return_value="Semantic tags describe meaning."):
            result = pipeline.answer_question("Explain it simply", mode="simple", history=history)
        called_query = mock_embed.call_args[0][0]
        assert "What are semantic tags in HTML?" in called_query
        assert "Explain it simply" in called_query
        assert result["error"] is None

    def test_answer_question_does_not_expand_standalone_short_query(self):
        from rag import pipeline
        df = make_df(5)
        pipeline._df = df
        history = [
            {"role": "user", "content": "What are semantic tags in HTML?"},
            {"role": "assistant", "content": "They give meaning to structure."},
        ]
        with patch("rag.pipeline.embed_query", return_value=df["embedding"].iloc[0]) as mock_embed, \
             patch("rag.pipeline.call_llm", return_value="Forms are in Video 7."):
            pipeline.answer_question("HTML forms", history=history)
        called_query = mock_embed.call_args[0][0]
        assert called_query == "HTML forms"

    def test_answer_question_below_threshold_returns_no_relevant_content(self):
        from rag import pipeline
        df = make_df(5, dim=4)
        pipeline._df = df
        with patch("rag.pipeline.embed_query", return_value=[1.0, 0.0, 0.0, 0.0]), \
             patch("rag.pipeline.find_top_k", return_value=(df.iloc[0:0], None)):
            result = pipeline.answer_question("What is the recipe for brownies?")
        assert result["error"] == "no_relevant_content"
        assert result["sources"] == []

    def test_get_all_videos_returns_list(self):
        from rag import pipeline
        pipeline._df = make_df()
        videos = pipeline.get_all_videos()
        assert isinstance(videos, list)
        assert all("number" in v and "title" in v for v in videos)

    def test_get_all_videos_sorted(self):
        from rag import pipeline
        pipeline._df = make_df(20)
        videos = pipeline.get_all_videos()
        numbers = [int(v["number"]) for v in videos]
        assert numbers == sorted(numbers)


# ── Quiz service tests ─────────────────────────────────────────────────────

class TestQuizService:

    def test_parse_quiz_json_valid(self):
        from services.quiz_service import _parse_quiz_json
        raw = json.dumps([{
            "question": "What is CSS?",
            "options": {"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
            "correct": "A",
            "explanation": "Because."
        }])
        result = _parse_quiz_json(raw)
        assert result is not None
        assert len(result) == 1

    def test_parse_quiz_json_with_markdown_fence(self):
        from services.quiz_service import _parse_quiz_json
        raw = "```json\n" + json.dumps([{
            "question": "Test?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct": "B",
            "explanation": "exp"
        }]) + "\n```"
        result = _parse_quiz_json(raw)
        assert result is not None

    def test_parse_quiz_json_with_object_wrapper(self):
        from services.quiz_service import _parse_quiz_json
        raw = json.dumps({
            "questions": [{
                "question": "What is CSS?",
                "options": {"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
                "correct": "A",
                "explanation": "Because."
            }]
        })
        result = _parse_quiz_json(raw)
        assert result is not None
        assert len(result) == 1

    def test_parse_quiz_json_invalid_json(self):
        from services.quiz_service import _parse_quiz_json
        result = _parse_quiz_json("not valid json at all ~~~")
        assert result is None

    def test_score_quiz_all_correct(self):
        from services.quiz_service import score_quiz
        questions = [
            {"question": "Q1?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "A", "explanation": "exp1"},
            {"question": "Q2?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "C", "explanation": "exp2"},
        ]
        answers = {"0": "A", "1": "C"}
        result = score_quiz(questions, answers)
        assert result["score"] == 2
        assert result["total"] == 2
        assert result["percentage"] == 100

    def test_score_quiz_all_wrong(self):
        from services.quiz_service import score_quiz
        questions = [
            {"question": "Q1?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "A", "explanation": ""},
        ]
        answers = {"0": "B"}
        result = score_quiz(questions, answers)
        assert result["score"] == 0
        assert result["percentage"] == 0

    def test_score_quiz_no_answer_given(self):
        from services.quiz_service import score_quiz
        questions = [
            {"question": "Q1?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "A", "explanation": ""},
        ]
        result = score_quiz(questions, {})
        assert result["score"] == 0

    def test_quiz_empty_topic(self):
        from services.quiz_service import generate_quiz
        result = generate_quiz("", num_questions=3)
        assert result["error"] == "empty_topic"

    def test_quiz_embeddings_not_loaded(self):
        from rag import pipeline
        pipeline._df = None
        from services.quiz_service import generate_quiz
        result = generate_quiz("CSS selectors", num_questions=3)
        assert result["error"] == "embeddings_not_loaded"


# ── Flask app tests ───────────────────────────────────────────────────────

class TestFlaskApp:

    @pytest.fixture(autouse=True)
    def setup_app(self):
        from rag import pipeline
        pipeline._df = make_df()
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test"
        self.client = flask_app.app.test_client()

    def test_health_endpoint(self):
        with patch("app.check_ai_health", return_value={
            "provider": "ollama",
            "ai_ready": True,
            "ollama_connected": True,
            "embed_model": "bge-m3",
            "llm_model": "llama3.2",
            "embed_model_ready": True,
            "llm_model_ready": True,
        }):
            r = self.client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert data["embeddings_loaded"] is True
        assert data["ollama_connected"] is True
        assert data["ai_ready"] is True

    def test_health_endpoint_gemini_mode_does_not_require_ollama(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
        with patch("rag.pipeline.requests.get") as mock_get:
            r = self.client.get("/health")
        mock_get.assert_not_called()
        assert r.status_code == 200
        data = r.get_json()
        assert data["provider"] == "gemini"
        assert data["embedding_provider"] == "gemini"
        assert data["llm_provider"] == "gemini"
        assert data["ai_ready"] is True
        assert data["ollama_connected"] is False

    def test_landing_page(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert b"LearnAI" in r.data or b"Teaching Assistant" in r.data

    def test_dashboard_page(self):
        r = self.client.get("/dashboard")
        assert r.status_code == 200

    def test_api_ask_empty_query(self):
        r = self.client.post("/api/ask",
                             json={"query": ""},
                             content_type="application/json")
        assert r.status_code == 400
        data = r.get_json()
        assert data["error"] == "empty_query"

    def test_api_ask_service_unavailable_returns_503(self):
        with patch("app.answer_question", return_value={
            "query": "What is HTML?",
            "mode": "default",
            "answer": "Ollama offline",
            "sources": [],
            "error": "embedding_failed",
        }):
            r = self.client.post("/api/ask",
                                 json={"query": "What is HTML?"},
                                 content_type="application/json")
        assert r.status_code == 503
        assert r.get_json()["error"] == "embedding_failed"

    def test_api_videos(self):
        r = self.client.get("/api/videos")
        assert r.status_code == 200
        data = r.get_json()
        assert "videos" in data
        assert isinstance(data["videos"], list)

    def test_api_history_empty(self):
        r = self.client.get("/api/history")
        assert r.status_code == 200
        data = r.get_json()
        assert "history" in data

    def test_api_clear_history(self):
        r = self.client.post("/api/history/clear")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "cleared"

    def test_api_quiz_empty_topic(self):
        r = self.client.post("/api/quiz/generate",
                             json={"topic": "", "num_questions": 3},
                             content_type="application/json")
        assert r.status_code == 400

    def test_api_quiz_invalid_num_questions(self):
        r = self.client.post("/api/quiz/generate",
                             json={"topic": "HTML", "num_questions": "abc"},
                             content_type="application/json")
        assert r.status_code == 400
        assert r.get_json()["error"] == "invalid_num_questions"

    def test_api_quiz_hides_answers_before_submission_and_scores_via_quiz_id(self):
        mock_quiz = {
            "topic": "HTML",
            "questions": [{
                "question": "What does HTML stand for?",
                "options": {"A": "HyperText Markup Language", "B": "Home Tool", "C": "Hyperlinks", "D": "None"},
                "correct": "A",
                "explanation": "HTML stands for HyperText Markup Language.",
            }],
            "sources": [],
            "error": None,
        }
        with patch("app.generate_quiz", return_value=mock_quiz):
            gen = self.client.post("/api/quiz/generate",
                                   json={"topic": "HTML", "num_questions": 1},
                                   content_type="application/json")
        assert gen.status_code == 200
        gen_data = gen.get_json()
        assert "quiz_id" in gen_data
        assert "correct" not in gen_data["questions"][0]
        assert "explanation" not in gen_data["questions"][0]

        scored = self.client.post("/api/quiz/score",
                                  json={"quiz_id": gen_data["quiz_id"], "answers": {"0": "A"}},
                                  content_type="application/json")
        assert scored.status_code == 200
        score_data = scored.get_json()
        assert score_data["score"] == 1
        assert score_data["percentage"] == 100

    def test_api_score_no_questions(self):
        r = self.client.post("/api/quiz/score",
                             json={"questions": [], "answers": {}},
                             content_type="application/json")
        assert r.status_code == 400

    def test_dotenv_loaded_in_app(self):
        from dotenv import dotenv_values
        env_example = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        vals = dotenv_values(env_example)
        assert vals["EMBEDDING_PROVIDER"] == "local"
        assert vals["LLM_PROVIDER"] == "gemini"
        assert vals["LOCAL_EMBEDDING_MODEL"] == "BAAI/bge-small-en-v1.5"
        assert vals["LOCAL_EMBEDDING_DIM"] == "384"
        assert vals["LOCAL_EMBEDDING_THREADS"] == "1"
        assert vals["GEMINI_GENERATION_MODEL"] == "gemini-3.5-flash-lite"
        assert "GEMINI_API_KEY" in vals
        assert vals["RELEVANCE_THRESHOLD"] == "0.68"
        assert vals["TOP_K"] == "5"
        assert vals["HOST"] == "0.0.0.0"
        assert "PORT" not in vals
        assert "GEMINI_EMBEDDING_MODEL" not in vals
        assert "GEMINI_EMBEDDING_DIM" not in vals
        assert vals["OLLAMA_BASE_URL"] == "http://localhost:11434"
        assert vals["EMBED_MODEL"] == "bge-m3"
        assert vals["LLM_MODEL"] == "llama3.2"

    def test_api_ask_rate_limited_returns_429(self):
        with patch("app.answer_question", return_value={
            "query": "What is HTML?",
            "mode": "default",
            "answer": "Rate limit exceeded.",
            "sources": [],
            "error": "rate_limited",
        }):
            r = self.client.post("/api/ask",
                                 json={"query": "What is HTML?"},
                                 content_type="application/json")
        assert r.status_code == 429
        assert r.get_json()["error"] == "rate_limited"


# ── Gemini provider & dual-provider tests ─────────────────────────────────

class TestGeminiProvider:

    def test_l2_normalize_unit_norm(self):
        from rag.gemini_client import l2_normalize
        vec = l2_normalize([3.0, 4.0])
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

    def test_gemini_embed_requires_api_key(self, monkeypatch):
        import config
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")

        from rag.retrieval import embed_query, get_last_embed_error
        vec = embed_query("What are semantic tags?")
        err_code, _ = get_last_embed_error()
        assert vec is None
        assert err_code == "gemini_not_configured"

    def test_gemini_embed_uses_task_type_and_normalizes(self, monkeypatch):
        import config
        from rag import gemini_client
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-unit-test")
        monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "4")
        gemini_client._client_cache.clear()

        mock_client = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [3.0, 4.0, 0.0, 0.0]
        mock_client.models.embed_content.return_value = MagicMock(embeddings=[mock_emb])

        with patch("rag.gemini_client.genai.Client", return_value=mock_client):
            from rag.retrieval import embed_query
            vec = embed_query("CSS box model")

        assert vec is not None
        assert len(vec) == 4
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == config.GEMINI_EMBEDDING_MODEL
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"
        assert call_kwargs["config"].output_dimensionality == 4

    def test_gemini_rate_limit_retries_and_succeeds(self, monkeypatch):
        from google.genai import errors
        from rag import gemini_client
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-unit-test")
        monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
        gemini_client._client_cache.clear()

        mock_client = MagicMock()
        rate_err = errors.APIError(429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
        ok_resp = MagicMock(text="Semantic tags give meaning to HTML structure.")
        mock_client.models.generate_content.side_effect = [rate_err, ok_resp]

        with patch("rag.gemini_client.genai.Client", return_value=mock_client), \
             patch("rag.gemini_client.time.sleep") as mock_sleep:
            from rag.generation import call_llm
            out = call_llm("Explain semantic tags")

        assert out == "Semantic tags give meaning to HTML structure."
        assert mock_client.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1

    def test_gemini_rate_limit_exhausted_returns_rate_limited(self, monkeypatch):
        from google.genai import errors
        from rag import gemini_client
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-unit-test")
        monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
        gemini_client._client_cache.clear()

        mock_client = MagicMock()
        rate_err = errors.APIError(429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
        mock_client.models.generate_content.side_effect = rate_err

        with patch("rag.gemini_client.genai.Client", return_value=mock_client), \
             patch("rag.gemini_client.time.sleep"):
            from rag.generation import call_llm, get_last_llm_error
            out = call_llm("Explain semantic tags")
            err_code, err_msg = get_last_llm_error()

        assert out is None
        assert err_code == "rate_limited"
        assert "fake-key" not in err_msg

    def test_gemini_auth_error_does_not_retry_or_leak_key(self, monkeypatch):
        from google.genai import errors
        from rag import gemini_client
        secret = "AIzaSySecretDoNotLeak12345"
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", secret)
        gemini_client._client_cache.clear()

        mock_client = MagicMock()
        auth_err = errors.APIError(403, {"error": {"message": f"Invalid key {secret}", "status": "PERMISSION_DENIED"}})
        mock_client.models.generate_content.side_effect = auth_err

        with patch("rag.gemini_client.genai.Client", return_value=mock_client):
            from rag.generation import call_llm, get_last_llm_error
            out = call_llm("hello")
            err_code, err_msg = get_last_llm_error()

        assert out is None
        assert err_code == "gemini_auth_error"
        assert secret not in err_msg
        assert mock_client.models.generate_content.call_count == 1

    def test_dimension_mismatch_handled_cleanly(self):
        from rag.retrieval import find_top_k, get_last_embed_error
        df = make_df(5, dim=16)
        res_df, scores = find_top_k(df, [0.1] * 8, top_k=3)
        err_code, _ = get_last_embed_error()
        assert res_df is None
        assert scores is None
        assert err_code == "embedding_dim_mismatch"

    def test_gemini_end_to_end_pipeline_and_quiz(self, monkeypatch):
        from rag import gemini_client, pipeline
        from services.quiz_service import generate_quiz

        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-unit-test")
        monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "4")
        gemini_client._client_cache.clear()

        df = pd.DataFrame([{
            "chunk_id": 0,
            "number": "11",
            "title": "Semantic Tags in HTML",
            "start": 40.0,
            "end": 65.0,
            "text": "Semantic tags give meaning to website structure like header, nav, and footer.",
            "embedding": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }])
        pipeline._df = df

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[1.0, 0.0, 0.0, 0.0])]
        )
        quiz_payload = json.dumps({
            "questions": [{
                "question": "Which tag is a semantic HTML element?",
                "options": {"A": "<header>", "B": "<div>", "C": "<span>", "D": "<b>"},
                "correct": "A",
                "explanation": "<header> describes page header structure.",
            }]
        })
        mock_client.models.generate_content.side_effect = [
            MagicMock(text="Semantic tags are covered in Video 11 at 00:40."),
            MagicMock(text=quiz_payload),
        ]

        with patch("rag.gemini_client.genai.Client", return_value=mock_client):
            ans = pipeline.answer_question("What are semantic tags?")
            quiz = generate_quiz("Semantic tags", num_questions=1)

        assert ans["error"] is None
        assert "Video 11" in ans["answer"]
        assert len(ans["sources"]) == 1
        assert ans["sources"][0]["video_number"] == "11"
        assert quiz["error"] is None
        assert len(quiz["questions"]) == 1
        assert quiz["questions"][0]["correct"] == "A"

    def test_ollama_starts_without_gemini_key(self, monkeypatch):
        import config
        from rag import pipeline
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDINGS_PATH", raising=False)
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")

        assert config.get_ai_provider() == "ollama"
        assert config.get_embedding_provider() == "ollama"
        assert config.get_llm_provider() == "ollama"
        assert config.get_embeddings_path().endswith("embeddings.joblib")
        assert pipeline.load_embeddings() is True

    def test_build_gemini_embeddings_helpers(self, tmp_path, monkeypatch):
        from pathlib import Path
        import build_gemini_embeddings as bge

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-unit-test")
        newjsons_dir = Path(os.path.dirname(__file__)).parent / "newjsons"

        # 1. Loads all 1,406 existing merged chunks deterministically
        chunks = bge.load_chunks(newjsons_dir)
        assert len(chunks) == 1406
        assert chunks[0]["number"] == "1"

        # 2. Refuses to overwrite original embeddings.joblib
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            bge.build_embeddings(
                input_dir=newjsons_dir,
                output_path=bge.config.BASE_DIR / "embeddings.joblib",
            )

        # 3. Validates schema, dim, and unit norm when writing to separate output
        mini_dir = tmp_path / "mini_jsons"
        mini_dir.mkdir()
        (mini_dir / "01_test.json").write_text(json.dumps({
            "chunks": [
                {"number": "1", "title": "Intro", "start": 0.0, "end": 10.0, "text": "Hello HTML"},
                {"number": "1", "title": "Intro", "start": 10.0, "end": 20.0, "text": "Hello CSS"},
            ]
        }), encoding="utf-8")

        out_file = tmp_path / "embeddings_gemini.joblib"
        monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "4")
        with patch("build_gemini_embeddings.gemini_embed_texts", return_value=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]):
            df = bge.build_embeddings(input_dir=mini_dir, output_path=out_file, batch_size=2, delay_seconds=0)

        assert out_file.exists()
        assert len(df) == 2
        assert list(df.columns) == ["number", "title", "start", "end", "text", "chunk_id", "embedding"]


# ── Local BGE-small + Hybrid (Local Embeddings + Gemini LLM) tests ────────

class TestLocalBGEAndHybridProvider:

    def test_split_provider_configuration(self, monkeypatch):
        import config

        # 1. Hybrid: local BGE embeddings + Gemini LLM
        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
        monkeypatch.delenv("EMBEDDINGS_PATH", raising=False)
        assert config.get_embedding_provider() == "local"
        assert config.get_llm_provider() == "gemini"
        assert config.get_embeddings_path().endswith("embeddings_local.joblib")
        assert config.get_relevance_threshold() == 0.68
        assert config.get_active_model_names() == ("BAAI/bge-small-en-v1.5", "gemini-3.5-flash-lite")

        # 2. Ollama fallback: both providers fall back cleanly
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        assert config.get_embedding_provider() == "ollama"
        assert config.get_llm_provider() == "ollama"
        assert config.get_embeddings_path().endswith("embeddings.joblib")
        assert config.get_relevance_threshold() == 0.55

    def test_preprocess_and_l2_normalize_consistency(self):
        from rag.local_embedder import BGE_QUERY_PREFIX, l2_normalize_matrix, preprocess_text

        doc = preprocess_text("  Semantic   tags   in HTML  ", is_query=False)
        qry = preprocess_text("  Semantic   tags   in HTML  ", is_query=True)
        assert doc == "Semantic tags in HTML"
        assert qry == f"{BGE_QUERY_PREFIX}Semantic tags in HTML"

        mat = l2_normalize_matrix(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        assert mat.dtype == np.float32
        assert abs(float(np.linalg.norm(mat[0])) - 1.0) < 1e-6
        assert float(np.linalg.norm(mat[1])) == 0.0

    def test_build_local_embeddings_protects_existing_files(self, tmp_path):
        from pathlib import Path
        import build_local_embeddings as ble

        newjsons_dir = Path(os.path.dirname(__file__)).parent / "newjsons"
        for protected in ("embeddings.joblib", "embeddings_gemini.joblib", "embeddings_gemini.checkpoint.joblib"):
            with pytest.raises(ValueError, match="Refusing to overwrite protected file"):
                ble.build_local_embeddings(
                    input_dir=newjsons_dir,
                    output_path=ble.config.BASE_DIR / protected,
                )

        mini_dir = tmp_path / "mini"
        mini_dir.mkdir()
        (mini_dir / "01_test.json").write_text(json.dumps({
            "chunks": [
                {"number": "1", "title": "Intro", "start": 0.0, "end": 5.0, "text": "HTML intro"},
                {"number": "1", "title": "Intro", "start": 5.0, "end": 10.0, "text": "CSS intro"},
            ]
        }), encoding="utf-8")

        fake_vecs = np.zeros((2, 384), dtype=np.float32)
        fake_vecs[0, 0] = 1.0
        fake_vecs[1, 1] = 1.0

        out_file = tmp_path / "embeddings_local.joblib"
        with patch("build_local_embeddings.local_embed_texts", return_value=fake_vecs):
            df = ble.build_local_embeddings(input_dir=mini_dir, output_path=out_file, batch_size=2)

        assert out_file.exists()
        assert len(df) == 2
        ble.validate_embeddings_df(df, expected_dim=384)

    def test_load_embeddings_caches_matrix_without_duplication(self, monkeypatch):
        from rag import pipeline

        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert pipeline.load_embeddings() is True
        df = pipeline.get_df()
        mat = pipeline.get_embedding_matrix()
        assert df is not None and mat is not None
        assert mat.shape == (1406, 384)
        assert mat.dtype == np.float32
        assert pipeline.get_embedding_matrix() is mat

    def test_real_local_bge_retrieval_and_recalibrated_threshold(self, monkeypatch):
        from rag import pipeline
        from rag.retrieval import embed_query, find_top_k

        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
        assert pipeline.load_embeddings() is True

        df = pipeline.get_df()
        mat = pipeline.get_embedding_matrix()

        # Covered query should exceed calibrated threshold (0.68) and hit Video 11
        q_vec = embed_query("What are semantic tags in HTML?")
        assert q_vec is not None
        assert len(q_vec) == 384
        assert abs(float(np.linalg.norm(q_vec)) - 1.0) < 1e-5
        top_df, sims = find_top_k(df, q_vec, top_k=5, embeddings_matrix=mat)
        assert top_df is not None and not top_df.empty
        assert str(top_df.iloc[0]["number"]) == "11"
        assert float(sims.max()) >= 0.68

        # Off-topic query should be filtered below 0.68
        off_vec = embed_query("What is the recipe for chocolate fudge brownies?")
        assert off_vec is not None and len(off_vec) == 384
        off_df, off_sims = find_top_k(df, off_vec, top_k=5, embeddings_matrix=mat)
        assert off_df is not None and off_df.empty
        assert float(off_sims.max()) < 0.68

    def test_hybrid_local_embeddings_with_mocked_gemini_qa_and_quiz(self, monkeypatch):
        from rag import gemini_client, pipeline
        from services.quiz_service import generate_quiz, score_quiz

        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-hybrid-test")
        monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
        gemini_client._client_cache.clear()
        assert pipeline.load_embeddings() is True

        mock_client = MagicMock()
        quiz_payload = json.dumps({
            "questions": [{
                "question": "Which HTML tag adds semantic meaning to a navigation bar?",
                "options": {"A": "<nav>", "B": "<div>", "C": "<span>", "D": "<b>"},
                "correct": "A",
                "explanation": "<nav> is a semantic tag for navigation links covered in Video 11.",
            }]
        })
        mock_client.models.generate_content.side_effect = [
            MagicMock(text="Semantic tags in HTML are taught in Video 11."),
            MagicMock(text=quiz_payload),
        ]

        with patch("rag.gemini_client.genai.Client", return_value=mock_client):
            ans = pipeline.answer_question("What are semantic tags in HTML?")
            quiz = generate_quiz("Semantic HTML tags", num_questions=1)

        assert ans["error"] is None
        assert "Video 11" in ans["answer"]
        assert len(ans["sources"]) == 5
        assert ans["sources"][0]["video_number"] == "11"

        assert quiz["error"] is None
        assert len(quiz["questions"]) == 1
        scored = score_quiz(quiz["questions"], {"0": "A"})
        assert scored["score"] == 1
        assert scored["percentage"] == 100

    def test_health_endpoint_hybrid_local_gemini_does_not_touch_ollama(self, monkeypatch):
        import app as flask_app
        from rag import pipeline

        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-health-test")
        assert pipeline.load_embeddings() is True

        client = flask_app.app.test_client()
        with patch("rag.pipeline.requests.get") as mock_get:
            r = client.get("/health")
        mock_get.assert_not_called()
        assert r.status_code == 200
        data = r.get_json()
        assert data["provider"] == "hybrid"
        assert data["embedding_provider"] == "local"
        assert data["llm_provider"] == "gemini"
        assert data["ai_ready"] is True
        assert data["embeddings_loaded"] is True
        assert data["total_chunks"] == 1406
        assert data["ollama_connected"] is False


