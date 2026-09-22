const config = window.CBT_STUDY_CONFIG || {};
const apiBase = String(config.apiBase || "").trim().replace(/\/$/, "");
const hasRemoteApi = /^https:\/\/[^/]+/i.test(apiBase);
const usesNgrok = hasRemoteApi && /(^|\.)ngrok(-free)?\.(app|dev)$/i.test(new URL(apiBase).hostname);

const storageKeys = { token: "cbt-live/token", participant: "cbt-live/participant" };
const state = {
  token: sessionStorage.getItem(storageKeys.token) || "",
  participant: sessionStorage.getItem(storageKeys.participant) || "",
  profiles: [],
  study: null,
  pollTimer: null,
  toastTimer: null,
  pendingActions: new Set()
};

const el = {
  views: {
    login: document.getElementById("login-view"),
    profiles: document.getElementById("profiles-view"),
    study: document.getElementById("study-view")
  },
  connection: document.getElementById("connection-status"),
  loginForm: document.getElementById("login-form"),
  participant: document.getElementById("participant-code"),
  access: document.getElementById("access-code"),
  loginButton: document.getElementById("login-button"),
  loginError: document.getElementById("login-error"),
  signOut: document.getElementById("sign-out-button"),
  home: document.getElementById("home-link"),
  profileGrid: document.getElementById("profile-grid"),
  back: document.getElementById("back-button"),
  studyName: document.getElementById("study-patient-name"),
  studyCondition: document.getElementById("study-condition"),
  studyNote: document.getElementById("study-note"),
  panelGrid: document.getElementById("panel-grid"),
  profileButton: document.getElementById("profile-button"),
  finishButton: document.getElementById("finish-button"),
  profileDialog: document.getElementById("profile-dialog"),
  closeProfile: document.getElementById("close-profile-button"),
  modalName: document.getElementById("modal-profile-name"),
  modalSummary: document.getElementById("modal-summary"),
  modalContext: document.getElementById("modal-context"),
  modalHistory: document.getElementById("modal-history"),
  modalCoping: document.getElementById("modal-coping"),
  modalGuidance: document.getElementById("modal-guidance"),
  toast: document.getElementById("toast")
};

function requestId() {
  return window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function showView(name) {
  Object.entries(el.views).forEach(([key, node]) => node.classList.toggle("hidden", key !== name));
  el.signOut.classList.toggle("hidden", !state.token);
}

function setConnection(mode, text) {
  el.connection.classList.remove("online", "offline");
  if (mode) el.connection.classList.add(mode);
  el.connection.querySelector("span").textContent = text;
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  el.toast.textContent = message;
  el.toast.classList.toggle("error", isError);
  el.toast.classList.remove("hidden");
  state.toastTimer = window.setTimeout(() => el.toast.classList.add("hidden"), 4500);
}

async function api(path, options = {}) {
  if (!hasRemoteApi) throw new Error("The public study API is not configured in config.js.");
  const headers = { ...(options.headers || {}) };
  if (usesNgrok) headers["ngrok-skip-browser-warning"] = "1";
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token && !options.skipAuth) headers.Authorization = `Bearer ${state.token}`;

  let response;
  try {
    response = await fetch(`${apiBase}${path}`, { ...options, headers });
  } catch (cause) {
    throw new Error("Could not reach the study server.", { cause });
  }
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = null; }
  if (!response.ok) {
    const error = new Error(payload?.detail || `Server error (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function checkHealth() {
  try {
    await api("/api/health", { skipAuth: true });
    setConnection("online", "Server online");
  } catch {
    setConnection("offline", "Server unavailable");
  }
}

function clearPoll() {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function signOut() {
  clearPoll();
  state.token = "";
  state.participant = "";
  state.profiles = [];
  state.study = null;
  state.pendingActions.clear();
  sessionStorage.removeItem(storageKeys.token);
  sessionStorage.removeItem(storageKeys.participant);
  el.access.value = "";
  el.panelGrid.replaceChildren();
  showView("login");
}

async function handleAuthenticatedError(error) {
  if (error.status === 401) {
    signOut();
    showToast("Your sign-in expired. Please enter your codes again.", true);
    return;
  }
  showToast(error.message, true);
}

function profileNumber(profile) {
  return String(Number(profile.id.split("-").pop()) || "").padStart(2, "0");
}

function renderProfiles() {
  el.profileGrid.replaceChildren();
  for (const profile of state.profiles) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "patient-card";
    const statusText = profile.study?.status === "finished" ? "Completed" : profile.study ? "In progress" : "Not started";
    const actionText = profile.study?.status === "finished" ? "Review conversations" : profile.study ? "Continue" : "Open profile";
    button.innerHTML = `
      <span class="patient-number">${profileNumber(profile)}</span>
      <h2>${escapeHtml(profile.display_name)}</h2>
      <span class="condition">${escapeHtml(profile.condition)}</span>
      <p>${escapeHtml(profile.short_description)}</p>
      <span class="study-state">${statusText}</span>
      <span class="card-action">${actionText} &rarr;</span>`;
    button.addEventListener("click", () => openPatient(profile));
    el.profileGrid.appendChild(button);
  }
}

async function loadProfiles() {
  clearPoll();
  try {
    const payload = await api("/api/profiles");
    state.profiles = payload.profiles;
    renderProfiles();
    showView("profiles");
  } catch (error) {
    handleAuthenticatedError(error);
  }
}

async function openPatient(profile) {
  try {
    const payload = await api("/api/studies", {
      method: "POST",
      body: JSON.stringify({ profile_id: profile.id })
    });
    state.study = payload.study;
    el.panelGrid.replaceChildren();
    renderStudy();
    showView("study");
    schedulePoll();
  } catch (error) {
    handleAuthenticatedError(error);
  }
}

function fillList(node, values) {
  node.replaceChildren();
  for (const value of values || []) {
    const item = document.createElement("li");
    item.textContent = value;
    node.appendChild(item);
  }
}

function openProfileDialog() {
  const profile = state.study?.profile;
  if (!profile) return;
  el.modalName.textContent = `${profile.display_name} · ${profile.condition}`;
  el.modalSummary.textContent = profile.summary;
  el.modalContext.textContent = profile.current_context;
  el.modalHistory.textContent = profile.relevant_history;
  fillList(el.modalCoping, profile.coping_strategies);
  fillList(el.modalGuidance, profile.role_guidance);
  el.profileDialog.showModal();
}

function panelStatus(panel) {
  if (state.study.status === "finished") return ["Finished", ""];
  if (panel.job?.status === "queued") {
    return [`Queued${panel.job.queue_position ? ` #${panel.job.queue_position}` : ""}`, "busy"];
  }
  if (panel.job?.status === "running") return ["Generating…", "busy"];
  if (panel.job?.status === "failed") return ["Response failed", "failed"];
  return [panel.messages.length ? "Ready" : "Not started", ""];
}

function messageSignature(panel) {
  return JSON.stringify({
    ids: panel.messages.map(message => message.id),
    job: panel.job && [panel.job.id, panel.job.status, panel.job.queue_position],
    start: panel.can_start,
    localPending: state.pendingActions.has(panel.id)
  });
}

function createPanel(panel) {
  const article = document.createElement("article");
  article.className = "chat-panel";
  article.dataset.panelId = panel.id;
  article.innerHTML = `
    <header class="panel-header">
      <div class="panel-person"><span class="panel-avatar">${escapeHtml(panel.label.slice(-1))}</span><h2>${escapeHtml(panel.label)}</h2></div>
      <span class="panel-status"></span>
    </header>
    <div class="messages" aria-live="polite"></div>
    <form class="composer">
      <textarea rows="1" maxlength="4000" aria-label="Respond as the patient" placeholder="Respond as the patient…"></textarea>
      <button class="send-button" type="submit" aria-label="Send">&#8593;</button>
    </form>`;
  const textarea = article.querySelector("textarea");
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 90)}px`;
    article.querySelector(".send-button").disabled = !textarea.value.trim() || textarea.disabled;
  });
  textarea.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      article.querySelector("form").requestSubmit();
    }
  });
  article.querySelector("form").addEventListener("submit", event => {
    event.preventDefault();
    sendPatientMessage(panel.id, textarea.value);
  });
  el.panelGrid.appendChild(article);
  return article;
}

function appendMessage(container, message) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${message.role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = message.role === "patient" ? "You" : "Therapist";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = message.content;
  wrapper.append(label, bubble);
  container.appendChild(wrapper);
}

function updatePanel(panel) {
  let article = el.panelGrid.querySelector(`[data-panel-id="${CSS.escape(panel.id)}"]`);
  if (!article) article = createPanel(panel);
  const [statusText, statusClass] = panelStatus(panel);
  const statusNode = article.querySelector(".panel-status");
  statusNode.textContent = statusText;
  statusNode.className = `panel-status ${statusClass}`.trim();

  const signature = messageSignature(panel);
  const messages = article.querySelector(".messages");
  if (messages.dataset.signature !== signature) {
    const wasAtBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 60;
    messages.replaceChildren();
    for (const message of panel.messages) appendMessage(messages, message);
    if (!panel.messages.length && panel.can_start && !state.pendingActions.has(panel.id)) {
      const empty = document.createElement("div");
      empty.className = "empty-panel";
      empty.innerHTML = `<div><p>Start when you are ready.</p><button class="primary-button" type="button">Start conversation</button></div>`;
      empty.querySelector("button").addEventListener("click", () => startPanel(panel.id));
      messages.appendChild(empty);
    }
    if (panel.job?.status === "queued" || panel.job?.status === "running" || state.pendingActions.has(panel.id)) {
      const pending = document.createElement("div");
      pending.className = "pending-row";
      pending.textContent = panel.job?.status === "queued"
        ? `Waiting in queue${panel.job.queue_position ? ` · position ${panel.job.queue_position}` : ""}`
        : "Therapist is responding…";
      messages.appendChild(pending);
    }
    if (panel.job?.status === "failed") {
      const failed = document.createElement("div");
      failed.className = "failed-row";
      failed.innerHTML = `<span>Response could not be generated.</span><button class="retry-button" type="button">Try again</button>`;
      failed.querySelector("button").addEventListener("click", () => retryPanel(panel.id));
      messages.appendChild(failed);
    }
    messages.dataset.signature = signature;
    if (wasAtBottom || panel.messages.length <= 2) requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
  }

  const textarea = article.querySelector("textarea");
  const send = article.querySelector(".send-button");
  textarea.disabled = !panel.can_send || state.pendingActions.has(panel.id);
  send.disabled = textarea.disabled || !textarea.value.trim();
}

function renderStudy() {
  const study = state.study;
  if (!study) return;
  el.studyName.textContent = study.profile.display_name;
  el.studyCondition.textContent = study.profile.condition;
  el.finishButton.disabled = study.status === "finished" || study.panels.some(panel => ["queued", "running"].includes(panel.job?.status));
  el.studyNote.textContent = study.status === "finished"
    ? "This session is complete. The conversations are available for review."
    : "Respond in character as the patient. You can move between conversations while replies wait in the queue.";
  for (const panel of study.panels) updatePanel(panel);
}

function hasPendingJobs() {
  return Boolean(state.study?.panels.some(panel => ["queued", "running"].includes(panel.job?.status)));
}

function schedulePoll() {
  clearPoll();
  if (!state.study || !hasPendingJobs()) return;
  state.pollTimer = window.setTimeout(refreshStudy, 1400);
}

async function refreshStudy() {
  if (!state.study) return;
  try {
    const payload = await api(`/api/studies/${state.study.id}`);
    state.study = payload.study;
    renderStudy();
    schedulePoll();
  } catch (error) {
    handleAuthenticatedError(error);
    if (state.study) state.pollTimer = window.setTimeout(refreshStudy, 3500);
  }
}

async function startPanel(panelId) {
  if (state.pendingActions.has(panelId)) return;
  state.pendingActions.add(panelId);
  renderStudy();
  try {
    await api(`/api/studies/${state.study.id}/panels/${panelId}/start`, {
      method: "POST",
      body: JSON.stringify({ client_request_id: requestId() })
    });
    await refreshStudy();
  } catch (error) {
    handleAuthenticatedError(error);
  } finally {
    state.pendingActions.delete(panelId);
    renderStudy();
  }
}

async function sendPatientMessage(panelId, rawContent) {
  const content = rawContent.trim();
  if (!content || state.pendingActions.has(panelId)) return;
  const article = el.panelGrid.querySelector(`[data-panel-id="${CSS.escape(panelId)}"]`);
  const textarea = article.querySelector("textarea");
  state.pendingActions.add(panelId);
  textarea.value = "";
  renderStudy();
  try {
    await api(`/api/studies/${state.study.id}/panels/${panelId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content, client_message_id: requestId() })
    });
    await refreshStudy();
  } catch (error) {
    textarea.value = content;
    handleAuthenticatedError(error);
  } finally {
    state.pendingActions.delete(panelId);
    renderStudy();
  }
}

async function retryPanel(panelId) {
  if (state.pendingActions.has(panelId)) return;
  state.pendingActions.add(panelId);
  renderStudy();
  try {
    await api(`/api/studies/${state.study.id}/panels/${panelId}/retry`, {
      method: "POST",
      body: JSON.stringify({ client_request_id: requestId() })
    });
    await refreshStudy();
  } catch (error) {
    handleAuthenticatedError(error);
  } finally {
    state.pendingActions.delete(panelId);
    renderStudy();
  }
}

async function finishStudy() {
  if (!state.study || state.study.status === "finished") return;
  if (!window.confirm("Finish this session and lock all six conversations?")) return;
  try {
    const payload = await api(`/api/studies/${state.study.id}/finish`, { method: "POST" });
    state.study = payload.study;
    renderStudy();
    showToast("Session finished.");
  } catch (error) {
    handleAuthenticatedError(error);
  }
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

el.loginForm.addEventListener("submit", async event => {
  event.preventDefault();
  el.loginError.classList.add("hidden");
  el.loginButton.disabled = true;
  try {
    const payload = await api("/api/auth/login", {
      method: "POST",
      skipAuth: true,
      body: JSON.stringify({ participant_code: el.participant.value.trim(), access_code: el.access.value })
    });
    state.token = payload.token;
    state.participant = payload.participant_code;
    sessionStorage.setItem(storageKeys.token, state.token);
    sessionStorage.setItem(storageKeys.participant, state.participant);
    await loadProfiles();
  } catch (error) {
    el.loginError.textContent = error.message;
    el.loginError.classList.remove("hidden");
  } finally {
    el.loginButton.disabled = false;
  }
});

el.signOut.addEventListener("click", signOut);
el.back.addEventListener("click", loadProfiles);
el.home.addEventListener("click", event => {
  event.preventDefault();
  if (state.token) loadProfiles();
});
el.profileButton.addEventListener("click", openProfileDialog);
el.closeProfile.addEventListener("click", () => el.profileDialog.close());
el.profileDialog.addEventListener("click", event => {
  if (event.target === el.profileDialog) el.profileDialog.close();
});
el.finishButton.addEventListener("click", finishStudy);

async function boot() {
  checkHealth();
  if (state.token) {
    el.participant.value = state.participant;
    await loadProfiles();
  } else {
    showView("login");
  }
}

boot();
