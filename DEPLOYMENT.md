# Deployment Guide — Hybrid Cloud Deployment (Render + Local Development)

This guide explains how to deploy **LearnAI (RAG-Based AI Teaching Assistant)** to **Render** using the hybrid provider architecture (**FastEmbed ONNX** for local retrieval embeddings + **Google Gemini API** for answer and quiz generation), as well as how to run the application locally.

> [!NOTE]
> This guide documents the target Render deployment configuration and post-deployment verification steps. Perform the checks in the **Post-Deployment Checks** section after deploying your service on Render to confirm live availability.

---

## 1. Architecture Overview

In production, retrieval runs in-process on CPU via lightweight ONNX embeddings while generative responses are handled by the Gemini API:

```text
[User Browser]
      │
      │ HTTPS
      ▼
[Render Web Service — Gunicorn (1 worker, 4 threads) -> Flask app:app]
      ├── Startup: loads embeddings_local.joblib (1,406 chunks × 384 dims, ~1.97 MB)
      ├── Query Embedding + Retrieval (In-Process CPU):
      │     FastEmbed ONNX (BAAI/bge-small-en-v1.5, 384 dims) -> Cosine Similarity -> Top-K Chunks
      └── Answer & Quiz Generation (HTTPS API):
            Google Gemini API (gemini-3.5-flash-lite) -> Grounded Response + Video Timestamps
```

### Why This Hybrid Design Fits Free-Tier Cloud Hosting
- **Lightweight Local Retrieval (`EMBEDDING_PROVIDER=local`):** Uses `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dimensions) on ONNX Runtime instead of heavy PyTorch/CUDA stacks. Combined with precomputed embeddings in `embeddings_local.joblib`, this avoids embedding API rate limits and is designed to minimize memory usage (local peak RSS reached ~362 MB), though Render Free's 512 MB memory limit is tight and Out-Of-Memory (OOM) restarts remain possible.
- **Cloud LLM Generation (`LLM_PROVIDER=gemini`):** Offloads language generation to the Google Gemini API over HTTPS so no local GPU or large LLM weights are needed on the web server.
- **Optional Local Fallback (`ollama`):** Ollama remains supported as an **optional local development fallback** (`AI_PROVIDER=ollama` or `EMBEDDING_PROVIDER=ollama` / `LLM_PROVIDER=ollama`), and is not required or used in Render cloud deployments.

---

## 2. Render Web Service Setup

### Pre-Deployment Repository Checklist
Verify the following exist at the root of your GitHub repository before creating the Render service:
- `app.py` exporting the Flask application object `app`
- `embeddings_local.joblib` (precomputed 384-dim embeddings for all 1,406 transcript chunks)
- `requirements.txt` containing runtime dependencies (ensure `gunicorn` is listed in `requirements.txt` so Render can execute the start command)
- `.gitignore` excluding `.env` so secrets are never pushed

### Create the Web Service on Render
1. Sign in to [Render](https://render.com/) and click **New + → Web Service**.
2. Connect your GitHub repository (`RAG-Based-AI-Teaching-Assistant`).
3. Configure the service settings:
   - **Runtime:** `Python`
   - **Instance Type:** `Free` (or higher if needed)
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn app:app --workers 1 --threads 4 --bind 0.0.0.0:$PORT --timeout 120
     ```

### Why These Gunicorn Flags Matter
- `--workers 1`: Prevents duplicating the FastEmbed ONNX runtime and embedding arrays across multiple processes to help stay within Render Free's tight 512 MB RAM limit, and keeps the in-memory quiz store (`_QUIZ_STORE`) consistent within a single process.
- `--threads 4`: Handles concurrent web requests cleanly within the single worker.
- `--bind 0.0.0.0:$PORT`: Binds to the dynamic port injected by Render at runtime.
- `--timeout 120`: Gives the worker sufficient headroom for cold starts, first-query FastEmbed ONNX model download/initialization, and Gemini API calls.

---

## 3. Environment Variables (Render Dashboard)

Add the following environment variables in **Render Dashboard → Your Service → Environment**. Do **not** set `PORT` manually — Render injects `PORT` automatically at runtime.

| Variable | Value | Notes |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | Uses FastEmbed ONNX against `embeddings_local.joblib` |
| `LLM_PROVIDER` | `gemini` | Uses Google Gemini API for answer and quiz generation |
| `GEMINI_API_KEY` | *(set your secret key in Render)* | Enter directly in Render's Environment settings; never commit |
| `GEMINI_GENERATION_MODEL` | `gemini-3.5-flash-lite` | Gemini model ID for generation |
| `RELEVANCE_THRESHOLD` | `0.68` | Calibrated cosine similarity threshold for `BAAI/bge-small-en-v1.5` |
| `TOP_K` | `5` | Number of transcript chunks retrieved per query |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the model used to build `embeddings_local.joblib` |
| `LOCAL_EMBEDDING_DIM` | `384` | Expected embedding dimensionality |
| `LOCAL_EMBEDDING_THREADS` | `1` | Caps ONNX Runtime threads to minimize memory overhead |
| `FLASK_DEBUG` | `false` | Disables debug mode and Werkzeug interactive debugger in production |
| `SECRET_KEY` | *(long random secret string)* | Required for Flask session signing in production |
| `COURSE_NAME` | `Sigma Web Development Course` | Course label displayed in the UI and API responses |

To generate a strong `SECRET_KEY` locally before pasting it into Render:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 4. Post-Deployment Checks

After Render finishes building and starting the service, run the following checks against your Render service URL (`https://<your-service>.onrender.com`):

1. **Health Endpoint (`GET /health`):**
   ```bash
   curl https://<your-service>.onrender.com/health
   ```
   Confirm HTTP `200` and verify the JSON response reports:
   - `"status": "ok"`
   - `"embeddings_loaded": true`
   - `"total_chunks": 1406`
   - `"videos": 18`
   - `"provider": "hybrid"`
   - `"embedding_provider": "local"`
   - `"llm_provider": "gemini"`
   - `"ai_ready": true`
   - `"gemini_configured": true`
   - `"embed_model": "BAAI/bge-small-en-v1.5"`
   - `"llm_model": "gemini-3.5-flash-lite"`

2. **Chat UI (`GET /`):**
   Open `https://<your-service>.onrender.com/` in a browser and confirm the chat page renders without asset errors.

3. **Dashboard (`GET /dashboard`):**
   Open `https://<your-service>.onrender.com/dashboard` and confirm course metadata and UI controls load cleanly.

4. **Videos API (`GET /api/videos`):**
   ```bash
   curl https://<your-service>.onrender.com/api/videos
   ```
   Confirm HTTP `200` and that all **18** Sigma Web Development Course videos are listed.

5. **End-to-End Query Check:**
   Submit a course question in the chat UI (e.g., *"What is the HTML box model or semantic tags?"*). On a freshly booted container, the first query may take extra time while FastEmbed downloads and loads `BAAI/bge-small-en-v1.5`. Subsequent queries use the cached model in memory and respond faster.

---

## 5. Free-Tier & Runtime Limitations

- **Inactivity Spin-Down & Cold Starts:** Render Free Web Services spin down after 15 minutes of inactivity. The first request after spin-down triggers a container cold start (typically around 1 minute), and the first semantic query re-initializes the local FastEmbed model. Continuous 24/7 instant uptime is not guaranteed on the Free tier.
- **512 MB RAM & Shared CPU Limits:** Memory is tightly constrained on Render Free. Measured local hybrid memory usage reached ~362 MB RSS, leaving limited headroom below the 512 MB limit; Out-Of-Memory (OOM) restarts remain possible under concurrent requests or runtime overhead. Keep Gunicorn at `--workers 1` and `LOCAL_EMBEDDING_THREADS=1`, and avoid installing heavy ML packages (such as `torch` or `sentence-transformers`).
- **Ephemeral Filesystem & In-Memory Quiz State:** Render Free instances do not provide persistent local disk storage across restarts or redeployments. FastEmbed re-downloads ONNX weights into the container cache after a cold restart, and in-memory quiz sessions (`_QUIZ_STORE`) are lost whenever the container restarts or spins down.
- **Gemini Free-Tier Rate Limits:** The Google Gemini API enforces per-minute and daily rate limits on free-tier keys. Under heavy traffic, requests may experience backoff retries or return rate-limit messages.

---

## 6. Security Practices

- **Never commit `.env` or secrets:** Ensure `.env` stays listed in `.gitignore` and never commit API keys, `SECRET_KEY` values, or credentials to Git.
- **Manage secrets in Render:** Enter `GEMINI_API_KEY` and `SECRET_KEY` directly in Render's Environment dashboard rather than storing them in repository files or build scripts.
- **Keep debug disabled:** Always run production with `FLASK_DEBUG=false`.
- **Rotate compromised keys:** If a key is ever exposed in terminal history, screenshots, or commits, revoke and regenerate it immediately in Google AI Studio.

---

## 7. Local Development Setup

To run the same hybrid stack locally:

1. **Create and activate a virtual environment, then install dependencies:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Create `.env` from `.env.example`:**
   ```bash
   # macOS / Linux:
   cp .env.example .env
   ```
   ```powershell
   # Windows PowerShell:
   Copy-Item .env.example .env
   ```

3. **Configure `.env`:**
   Set your `GEMINI_API_KEY` and a local `SECRET_KEY`. For local-only binding on your machine, you may set `HOST=127.0.0.1` in `.env`. (When `PORT` is unset locally, `app.py` defaults to `5000`.)

4. **Start the Flask server:**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` (or `http://localhost:5000`) and verify `http://127.0.0.1:5000/health`.

### Optional Local Ollama Fallback
For completely offline local development without Gemini, install [Ollama](https://ollama.com/), pull `bge-m3` and `llama3.2`, and override the providers in your local `.env`:
```env
AI_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
LLM_PROVIDER=ollama
```
Ensure `embeddings.joblib` (generated with `bge-m3`) is present in the project root when running in full `ollama` mode.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `gunicorn: command not found` on Render | `gunicorn` missing from `requirements.txt` | Add `gunicorn` to `requirements.txt`, commit, and redeploy |
| Render worker killed / OOM (`502` or restart loop) | Too many Gunicorn workers or high ONNX thread count | Keep `--workers 1` and `LOCAL_EMBEDDING_THREADS=1` |
| First `/api/ask` request times out | Cold-start FastEmbed ONNX model download/load exceeded timeout | Ensure `--timeout 120` in the Gunicorn start command and retry |
| `/health` shows `"embeddings_loaded": false` | `embeddings_local.joblib` missing from repo root or wrong dimension | Ensure `embeddings_local.joblib` is committed at the root and `LOCAL_EMBEDDING_DIM=384` |
| Gemini errors / `ai_ready: false` | Missing/invalid `GEMINI_API_KEY` or wrong model name | Verify `GEMINI_API_KEY` and `GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite` in Render Environment settings |
| Rate-limit errors on `/api/ask` or `/api/quiz/generate` | Gemini API free-tier quota reached | Wait for rate-limit window reset or upgrade API quota |
