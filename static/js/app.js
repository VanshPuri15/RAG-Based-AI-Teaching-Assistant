/* ══════════════════════════════════════════════════════════════
   AI Teaching Assistant — Main Application JS
   ══════════════════════════════════════════════════════════════ */

"use strict";

/* ── Utilities ────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);
const qsa = sel => document.querySelectorAll(sel);

function escHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function fmtTime(seconds) {
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/* ══════════════════════════════════════════════════════════════
   SIDEBAR NAVIGATION
   ══════════════════════════════════════════════════════════════ */
const sidebar = document.querySelector(".sidebar");
const overlay = $("sidebarOverlay");

function openSidebar() {
  sidebar?.classList.add("open");
  overlay?.classList.add("open");
}
function closeSidebar() {
  sidebar?.classList.remove("open");
  overlay?.classList.remove("open");
}

$("sidebarOpen")?.addEventListener("click", openSidebar);
$("sidebarClose")?.addEventListener("click", closeSidebar);
overlay?.addEventListener("click", closeSidebar);

// Panel switching
let activePanel = "chat";
function switchPanel(name) {
  qsa(".panel").forEach(p => p.classList.add("d-none"));
  const target = $(`panel-${name}`);
  if (target) target.classList.remove("d-none");

  qsa(".sidebar-link").forEach(l => l.classList.remove("active"));
  const link = qs(`.sidebar-link[data-panel="${name}"]`);
  if (link) link.classList.add("active");

  activePanel = name;
  closeSidebar();

  if (name === "history") loadHistory();
}

qsa(".sidebar-link").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    switchPanel(link.dataset.panel);
  });
});

/* ══════════════════════════════════════════════════════════════
   OLLAMA STATUS CHECK
   ══════════════════════════════════════════════════════════════ */
async function checkOllamaStatus() {
  const dot = $("ollamaStatus");
  if (!dot) return;
  try {
    const r = await fetch("/health", { signal: AbortSignal.timeout(5000) });
    const data = await r.json();
    const llmProv = data.llm_provider || data.provider;
    const embProv = data.embedding_provider || data.provider;
    const providerLabel = (embProv === "local" && llmProv === "gemini")
      ? "Local BGE + Gemini API"
      : (llmProv === "gemini" ? "Gemini API" : "Ollama");
    const ready = data.ai_ready !== undefined ? data.ai_ready : data.ollama_connected;
    if (data.status === "ok" && data.embeddings_loaded && ready) {
      dot.className = "status-dot online";
      dot.title = `${providerLabel} connected · ${data.total_chunks} chunks loaded`;
    } else if (data.embeddings_loaded && !ready) {
      dot.className = "status-dot offline";
      dot.title = llmProv === "gemini"
        ? "Gemini API key not configured on server"
        : "Ollama offline — start Ollama locally to ask questions";
    } else {
      dot.className = "status-dot offline";
      dot.title = "Embeddings not loaded";
    }
  } catch {
    dot.className = "status-dot offline";
    dot.title = "Cannot reach server";
  }
}
checkOllamaStatus();

/* ══════════════════════════════════════════════════════════════
   CHAT
   ══════════════════════════════════════════════════════════════ */
let currentMode = "default";
let isWaiting = false;

// Mode selector
qsa(".mode-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    qsa(".mode-pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    currentMode = pill.dataset.mode;
  });
});

// Auto-resize textarea
const chatInput = $("chatInput");
chatInput?.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

// Enter to send (Shift+Enter for newline)
chatInput?.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

$("sendBtn")?.addEventListener("click", sendMessage);

// Suggested questions
qsa(".suggested-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (chatInput) chatInput.value = btn.dataset.q;
    sendMessage();
  });
});

// Clear chat
$("clearChatBtn")?.addEventListener("click", async () => {
  if (!confirm("Clear the chat history?")) return;
  await fetch("/api/history/clear", { method: "POST" });
  const msgs = $("chatMessages");
  msgs.innerHTML = "";
  renderWelcome();
});

function renderWelcome() {
  const msgs = $("chatMessages");
  msgs.innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon"><i class="bi bi-mortarboard-fill"></i></div>
      <h5 class="fw-600 mb-2">Hello! I'm your AI Course Assistant</h5>
      <p class="text-muted mb-4">
        I can answer questions about the <strong>${escHtml(typeof COURSE_NAME !== 'undefined' ? COURSE_NAME : 'Sigma Web Development Course')}</strong>.
        I'll always tell you exactly which video and timestamp covers each topic.
      </p>
      <div class="suggested-qs">
        <div class="suggested-label">Try asking:</div>
        <div class="d-flex flex-wrap gap-2 justify-content-center">
          ${["What are semantic HTML tags?","How do CSS selectors work?","Explain the CSS box model","What is the difference between inline and block elements?","How do I create an HTML form?"]
            .map(q => `<button class="suggested-btn" data-q="${escHtml(q)}">${escHtml(q)}</button>`).join("")}
        </div>
      </div>
    </div>`;
  // Re-attach click handlers for new suggested buttons
  msgs.querySelectorAll(".suggested-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (chatInput) chatInput.value = btn.dataset.q;
      sendMessage();
    });
  });
}

async function sendMessage() {
  if (isWaiting) return;
  const query = chatInput.value.trim();
  if (!query) return;

  chatInput.value = "";
  chatInput.style.height = "auto";

  // Hide welcome if visible
  qs(".chat-welcome")?.remove();

  appendUserBubble(query);
  const typingEl = appendTypingIndicator();

  isWaiting = true;
  $("sendBtn").disabled = true;

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: currentMode }),
    });
    const data = await r.json();
    typingEl.remove();
    appendAIResponse(data);
  } catch (err) {
    typingEl.remove();
    appendAIResponse({
      answer: "⚠️ Could not reach the server. Please make sure Flask is running.",
      sources: [],
      error: "network",
    });
  } finally {
    isWaiting = false;
    $("sendBtn").disabled = false;
    chatInput.focus();
  }
}

function appendUserBubble(text) {
  const msgs = $("chatMessages");
  const turn = document.createElement("div");
  turn.className = "chat-turn user";
  turn.innerHTML = `<div class="chat-bubble">${escHtml(text)}</div>`;
  msgs.appendChild(turn);
  scrollToBottom();
}

function appendTypingIndicator() {
  const msgs = $("chatMessages");
  const wrap = document.createElement("div");
  wrap.className = "chat-turn ai";
  wrap.innerHTML = `<div class="typing-indicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>`;
  msgs.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function appendAIResponse(data) {
  const msgs = $("chatMessages");
  const turn = document.createElement("div");
  turn.className = "chat-turn ai";

  // Format answer text — convert **bold** and `code` markdown patterns
  const formatted = formatAnswer(data.answer || "No response.");

  let html = `<div class="chat-bubble">${formatted}</div>`;

  // Source cards
  if (data.sources && data.sources.length > 0) {
    const sourceHtml = data.sources
      .map(s => `
        <div class="source-card">
          <div class="source-card-header">
            <i class="bi bi-play-circle-fill"></i>
            <span>Video ${escHtml(s.video_number)} · ${escHtml(s.title)}</span>
          </div>
          <div class="source-card-ts">
            <i class="bi bi-clock"></i>
            <span class="ts-chip">${escHtml(s.start_fmt)} – ${escHtml(s.end_fmt)}</span>
            <span class="text-muted">· Similarity: ${(s.similarity * 100).toFixed(0)}%</span>
          </div>
          <div class="source-card-excerpt">"${escHtml(s.excerpt)}"</div>
        </div>
      `).join("");

    html += `
      <div class="source-cards mt-2">
        <div class="suggested-label mb-2"><i class="bi bi-bookmark-fill me-1 text-accent-teal"></i>Relevant sources</div>
        ${sourceHtml}
      </div>`;
  }

  turn.innerHTML = html;
  msgs.appendChild(turn);
  scrollToBottom();
}

function formatAnswer(text) {
  // Escape HTML first
  let s = escHtml(text);
  // **bold** → <strong>
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // `code` → <code>
  s = s.replace(/`(.+?)`/g, "<code>$1</code>");
  // newlines → <br>
  s = s.replace(/\n/g, "<br>");
  return s;
}

function scrollToBottom() {
  const msgs = $("chatMessages");
  msgs.scrollTop = msgs.scrollHeight;
}

/* ══════════════════════════════════════════════════════════════
   HISTORY PANEL
   ══════════════════════════════════════════════════════════════ */
async function loadHistory() {
  try {
    const r = await fetch("/api/history");
    const data = await r.json();
    renderHistory(data.history || []);
  } catch {
    $("historyList").innerHTML = `<p class="text-muted">Could not load history.</p>`;
  }
}

function renderHistory(history) {
  const list = $("historyList");
  if (!history.length) {
    list.innerHTML = `<div class="text-center text-muted py-5">
      <i class="bi bi-clock-history fs-2 d-block mb-2 opacity-50"></i>
      <p>No conversation history yet.</p>
      <p class="small">Ask a question in the AI Assistant tab to get started.</p>
    </div>`;
    return;
  }
  list.innerHTML = history.map((msg, i) => `
    <div class="history-item">
      <div class="history-role ${msg.role}">${msg.role === "user" ? "You" : "AI Assistant"}</div>
      <div class="small">${escHtml(msg.content).replace(/\n/g, "<br>")}</div>
    </div>
  `).join("");
}

$("clearHistoryBtn")?.addEventListener("click", async () => {
  if (!confirm("Clear conversation history?")) return;
  await fetch("/api/history/clear", { method: "POST" });
  renderHistory([]);
});

/* ══════════════════════════════════════════════════════════════
   QUIZ
   ══════════════════════════════════════════════════════════════ */
let quizData = null;  // { questions, sources, topic }

// Num questions slider
const numQSlider = $("numQuestions");
const numQLabel = $("numQLabel");
numQSlider?.addEventListener("input", () => { numQLabel.textContent = numQSlider.value; });

// Topic chips
qsa(".topic-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    $("quizTopic").value = chip.dataset.topic;
  });
});

$("generateQuizBtn")?.addEventListener("click", generateQuiz);

async function generateQuiz() {
  const topic = $("quizTopic").value.trim();
  const numQ = parseInt($("numQuestions").value);
  const errEl = $("quizError");

  if (!topic) {
    errEl.textContent = "Please enter a topic.";
    errEl.classList.remove("d-none");
    return;
  }
  errEl.classList.add("d-none");

  const btn = $("generateQuizBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Generating quiz...`;

  try {
    const r = await fetch("/api/quiz/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, num_questions: numQ }),
    });
    const data = await r.json();

    if (data.error && data.error !== "no_relevant_content") {
      throw new Error(data.error_message || data.error);
    }
    if (!data.questions || data.questions.length === 0) {
      errEl.textContent = data.error_message || "Could not generate quiz on this topic. Try a different topic from the course.";
      errEl.classList.remove("d-none");
      return;
    }

    quizData = data;
    renderQuizQuestions(data);
  } catch (err) {
    errEl.textContent = err.message || "Failed to generate quiz. Make sure Ollama is running.";
    errEl.classList.remove("d-none");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="bi bi-stars me-2"></i>Generate Quiz`;
  }
}

function renderQuizQuestions(data) {
  $("quizSetup").classList.add("d-none");
  $("quizResults").classList.add("d-none");
  $("quizQuestions").classList.remove("d-none");

  $("quizTopicDisplay").textContent = `Quiz: ${data.topic}`;
  $("quizProgress").textContent = `${data.questions.length} questions`;
  $("quizProgressBar").style.width = "0%";

  const container = $("questionsContainer");
  container.innerHTML = data.questions.map((q, i) => `
    <div class="quiz-question-card" id="qcard-${i}">
      <div class="quiz-q-num">Question ${i + 1} of ${data.questions.length}</div>
      <div class="quiz-q-text">${escHtml(q.question)}</div>
      <div class="quiz-options" id="opts-${i}">
        ${Object.entries(q.options).map(([letter, text]) => `
          <div class="quiz-option" data-q="${i}" data-opt="${escHtml(letter)}">
            <div class="option-letter">${escHtml(letter)}</div>
            <div class="option-text">${escHtml(text)}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `).join("");

  container.querySelectorAll(".quiz-option").forEach(optEl => {
    optEl.addEventListener("click", () => {
      const qIdx = parseInt(optEl.dataset.q, 10);
      const letter = optEl.dataset.opt;
      selectOption(qIdx, letter, optEl);
    });
  });

  // Track selections
  window._quizSelections = {};
  updateQuizProgress();
}

function selectOption(qIdx, letter, el) {
  // Clear previous selection for this question
  document.querySelectorAll(`#opts-${qIdx} .quiz-option`).forEach(o => o.classList.remove("selected"));
  el.classList.add("selected");
  window._quizSelections[qIdx] = letter;
  updateQuizProgress();
}

function updateQuizProgress() {
  if (!quizData) return;
  const answered = Object.keys(window._quizSelections || {}).length;
  const total = quizData.questions.length;
  const pct = Math.round((answered / total) * 100);
  $("quizProgressBar").style.width = pct + "%";
  $("quizProgress").textContent = `${answered}/${total} answered`;
}

$("submitQuizBtn")?.addEventListener("click", async () => {
  if (!quizData) return;

  const unanswered = quizData.questions.length - Object.keys(window._quizSelections || {}).length;
  if (unanswered > 0 && !confirm(`You have ${unanswered} unanswered question(s). Submit anyway?`)) return;

  try {
    const r = await fetch("/api/quiz/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quiz_id: quizData.quiz_id,
        answers: window._quizSelections || {},
      }),
    });
    const result = await r.json();
    renderQuizResults(result);
  } catch {
    alert("Failed to score quiz. Please try again.");
  }
});

$("newQuizBtn")?.addEventListener("click", resetQuiz);
$("tryAgainBtn")?.addEventListener("click", resetQuiz);

function resetQuiz() {
  quizData = null;
  window._quizSelections = {};
  $("quizSetup").classList.remove("d-none");
  $("quizQuestions").classList.add("d-none");
  $("quizResults").classList.add("d-none");
  $("quizTopic").value = "";
  $("quizError").classList.add("d-none");
  $("numQuestions").value = 5;
  $("numQLabel").textContent = "5";
}

function renderQuizResults(result) {
  $("quizQuestions").classList.add("d-none");
  $("quizResults").classList.remove("d-none");

  const pct = result.percentage;
  const circleEl = $("scoreCircle");
  circleEl.style.background = `conic-gradient(${pct >= 70 ? 'var(--accent-green)' : pct >= 40 ? 'var(--accent-blue)' : 'var(--accent-red)'} ${pct}%, var(--brand-border) 0%)`;
  $("scoreDisplay").textContent = `${pct}%`;

  const msg = pct >= 80 ? "Excellent work! 🎉" : pct >= 60 ? "Good job! 👍" : pct >= 40 ? "Keep studying! 📚" : "Review the material and try again! 💪";
  $("scoreMessage").textContent = msg;
  $("scoreSub").textContent = `You got ${result.score} out of ${result.total} questions correct.`;

  const detail = $("resultsDetail");
  detail.innerHTML = `<h6 class="fw-600 mb-3">Detailed Results</h6>` +
    result.results.map((r, i) => `
      <div class="result-item ${r.is_correct ? 'correct' : 'incorrect'}">
        <div class="d-flex align-items-center gap-2 mb-1">
          <i class="bi ${r.is_correct ? 'bi-check-circle-fill text-accent-green' : 'bi-x-circle-fill text-accent-red'}"></i>
          <span class="fw-600 small">Q${i + 1}: ${escHtml(r.question)}</span>
        </div>
        <div class="small">
          Your answer: <span class="ts-chip">${r.chosen || "—"}</span>
          Correct: <span class="ts-chip" style="background:rgba(78,205,196,.15);color:var(--accent-teal)">${r.correct}</span>
        </div>
        <div class="result-explanation">💡 ${escHtml(r.explanation)}</div>
      </div>
    `).join("");
}

/* ══════════════════════════════════════════════════════════════
   LANDING PAGE (index.html) — no dashboard elements, skip
   ══════════════════════════════════════════════════════════════ */
// No special init needed for landing
