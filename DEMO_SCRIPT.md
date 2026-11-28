# FLARE Hackathon — 3-Minute Demo Script & Submission Pack

---

## Project Submission Summary (for FLARE Form)

### Project Name
**LearnAI — RAG-Based AI Teaching Assistant**

### Concise Project Description
LearnAI is a personal AI-powered learning companion built on Retrieval-Augmented Generation (RAG) that grounds every explanation, source reference, and practice quiz in actual course video transcripts—showing students not only *what* the answer is, but *which lesson* and *exact timestamp* teaches it.

### Problem Statement
Students learning from long technical video playlists waste significant time scrubbing through hours of footage to locate a single explanation. Meanwhile, generic AI chatbots frequently give generic or off-syllabus answers without pointing learners back to their actual course material.

### Solution Description
LearnAI indexes timestamped video transcripts using **BGE-M3** embeddings and retrieves the most relevant course segments via cosine similarity before generating grounded responses with **Llama 3.2** through **Ollama**. Built with a modular **Flask** backend and a responsive web UI, it provides:
- Timestamped source reference cards extracted directly from course metadata.
- Five explanation styles (*Standard*, *Beginner*, *Step-by-step*, *Summary*, *Example*).
- Relevance-threshold guardrails that refuse to hallucinate course citations for unrelated questions.
- A course-grounded AI Quiz Generator with server-side answer hiding and instant scoring.

### Key Innovation Points
1. **Timestamp-Level Grounding from Real Metadata:** Source cards (`Video #`, `Title`, `MM:SS – MM:SS`, `Similarity %`) come strictly from retrieved DataFrame metadata rather than LLM guesses.
2. **Pedagogical Modes + Honest Guardrails:** Students choose *how* they want concepts explained while out-of-scope queries are filtered by similarity thresholding and untrusted-data prompt framing.
3. **100% Local-First Privacy & Zero Inference Cost:** Runs end-to-end on local **BGE-M3** + **Llama 3.2** via Ollama—requiring zero paid API keys.

---

## 3-Minute Live Demo Script

> **Pre-demo check (30 seconds before presenting):**
> 1. Confirm Ollama is running with `bge-m3` and `llama3.2`.
> 2. Confirm `python app.py` is running and `http://127.0.0.1:5000/health` returns `"ollama_connected": true`.
> 3. Open `/` in one tab and `/dashboard` in another.

---

### 0:00 – 0:30 | 1. The Problem
> *"Imagine you're learning web development from an 18-video course and forget where **Semantic HTML tags** or the **CSS Box Model** were explained. Scrubbing through 30-minute videos is slow—and asking a generic AI chatbot gives generic answers with no link back to your instructor's lesson.*
>
> *We built **LearnAI**—an AI Teaching Assistant grounded directly in the actual course transcripts."*

*(Show the Landing Page at `/` briefly—highlight the 18 indexed videos, 1,406 transcript chunks, and click **Start Learning**.)*

---

### 0:30 – 1:15 | 2. Live Question & 3. How RAG Retrieves Content
> *"Let's open the Learning Dashboard. On the left, all 18 lessons from the Sigma Web Development Course are indexed.*
>
> *I'll ask a real question: **'What are semantic tags in HTML?'***"

*(Click the suggested question or type it and press Send.)*

> *"While it generates, here's what happens behind the scenes:*
> 1. *Our Flask backend sends the question to **Ollama's BGE-M3** model to generate a 1024-dimensional embedding.*
> 2. *It runs **cosine similarity** across all 1,406 precomputed transcript chunks stored in `embeddings.joblib`.*
> 3. *If the top similarity falls below our threshold (`0.55`), it refuses to fabricate course citations. Otherwise, the top 5 relevant transcript segments are passed to **Llama 3.2** inside untrusted-data delimiters."*

---

### 1:15 – 1:55 | 4. Grounded Answer + Video Timestamps & Sources
> *"Look at the response:*
> - *The explanation comes directly from the course material.*
> - *Right below the answer, **Relevant Sources** displays the exact metadata from our retrieval index—for example, **Video 11 · Semantic Tags in HTML**, timestamps like **`07:48 – 08:11`**, similarity percentage, and the transcript excerpt.*
> - *Notice the timestamps aren't guessed by the LLM—they're pulled straight from our indexed metadata."*

*(Optional quick demo if time permits: click **Beginner** mode and ask a short follow-up like `"Give me a simple analogy for it"` or show that session history preserves context.)*

---

### 1:55 – 2:35 | 5. Additional Learning Feature — AI Quiz Generator
> *"Learning isn't passive, so we also added a **Quiz Generator** grounded in the same RAG retrieval pipeline.*
>
> *Let's switch to **Quiz Generator**, pick **CSS selectors** (3 questions), and click **Generate Quiz**.*
> - *The backend retrieves the top matching course chunks and generates JSON-validated multiple-choice questions strictly from what was taught.*
> - *Correct answers stay hidden on the server until submission—never exposed in the browser response.*
> - *When I submit, it scores my answers and gives explanations."*

*(Select answers, click **Submit Quiz**, and show the score card + explanations.)*

---

### 2:35 – 3:00 | 6. Real-World Impact & Feasibility
> *"Why does this matter? Any educator, university, or EdTech platform with video lectures can run our pipeline—extract audio with FFmpeg, transcribe with Whisper, index with BGE-M3, and launch a private, zero-API-cost AI tutor for their students.*
>
> *By combining accurate semantic retrieval with exact lesson timestamps and active quiz recall, students save hours of video scrubbing and actually retain what they watch. Thank you!"*

---

## Potential Judge Questions & Answers

**Q1: How do you prevent hallucinations or fake timestamps?**
> Two safeguards: (1) A cosine-similarity `RELEVANCE_THRESHOLD` (`0.55` by default) checks whether any transcript chunk actually matches the query before asking the LLM to answer from course content; and (2) the Source Reference cards rendered below every answer come directly from the retrieved `embeddings.joblib` DataFrame metadata (`number`, `title`, `start`, `end`), not from LLM-generated text.

**Q2: Why run BGE-M3 and Llama 3.2 locally via Ollama instead of a cloud API?**
> Privacy, reproducibility, and zero per-token cost. Student questions and proprietary lecture transcripts never leave the host machine, and our config (`OLLAMA_BASE_URL`, `EMBED_MODEL`, `LLM_MODEL` in `.env`) keeps model endpoints decoupled so switching to a remote host or hosted adapter later requires minimal change.

**Q3: What happens if retrieved transcripts contain prompt-injection text or an off-topic question is asked?**
> In `rag/generation.py` and `services/quiz_service.py`, retrieved transcript text is wrapped in `<transcript_data>` delimiters with an explicit system rule to treat transcripts strictly as untrusted data. Off-topic questions below the similarity threshold trigger an empty retrieval set so the assistant politely informs the student the topic isn't covered in the course.

**Q4: How do you make sure students can't inspect quiz answers in DevTools before submitting?**
> `/api/quiz/generate` stores the full quiz (including `correct` and `explanation`) in a server-side bounded store keyed by `quiz_id`, stripping `correct` and `explanation` before returning the questions JSON to the browser. Scoring happens server-side in `/api/quiz/score`.

**Q5: How would this scale to 500+ videos?**
> Currently, 1,406 chunks (18 videos) search in milliseconds using NumPy/Scikit-learn cosine similarity in memory. For 50,000+ chunks, the `find_top_k` interface in `rag/retrieval.py` can be swapped to FAISS, Qdrant, or pgvector without changing Flask routes or the frontend.
