const config = window.CBT_STUDY_CONFIG || {};
const apiBase = String(config.apiBase || "").trim().replace(/\/$/, "");
const hasRemoteApi = /^https:\/\/[^/]+/i.test(apiBase);
const usesNgrok = hasRemoteApi && /(^|\.)ngrok(-free)?\.(app|dev)$/i.test(new URL(apiBase).hostname);
const ctrsScale = window.CTRS_SCALE || [];
const ctrsRubric = window.CTRS_RUBRIC || [];

const storageKeys = { token: "cbt-live/token", participant: "cbt-live/participant" };
const state = {
  token: sessionStorage.getItem(storageKeys.token) || "",
  participant: sessionStorage.getItem(storageKeys.participant) || "",
  profiles: [],
  study: null,
  view: "login",
  pollTimer: null,
  toastTimer: null,
  ratingPanelId: null,
  pendingActions: new Set()
};

const el = {
  views: {
    login: document.getElementById("login-view"),
    profiles: document.getElementById("profiles-view"),
    patient: document.getElementById("patient-view"),
    session: document.getElementById("session-view"),
    rating: document.getElementById("rating-view")
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
  patientBack: document.getElementById("patient-back-button"),
  patientName: document.getElementById("patient-name"),
  patientCondition: document.getElementById("patient-condition"),
  patientProgress: document.getElementById("patient-progress"),
  patientSummary: document.getElementById("patient-summary"),
  patientContext: document.getElementById("patient-context"),
  patientHistory: document.getElementById("patient-history"),
  patientCoping: document.getElementById("patient-coping"),
  patientGuidance: document.getElementById("patient-guidance"),
  patientNextNote: document.getElementById("patient-next-note"),
  beginSession: document.getElementById("begin-session-button"),
  sessionBack: document.getElementById("session-back-button"),
  sessionProgress: document.getElementById("session-progress"),
  sessionTherapist: document.getElementById("session-therapist"),
  profileButton: document.getElementById("profile-button"),
  endSession: document.getElementById("end-session-button"),
  panelHost: document.getElementById("panel-host"),
  ratingProgress: document.getElementById("rating-progress"),
  ratingTitle: document.getElementById("rating-title"),
  automaticEndNote: document.getElementById("automatic-end-note"),
  ratingProfileButton: document.getElementById("rating-profile-button"),
  ratingTranscript: document.getElementById("rating-transcript"),
  ratingForm: document.getElementById("rating-form"),
  ratingItems: document.getElementById("rating-items"),
  ratingComments: document.getElementById("rating-comments"),
  ratingError: document.getElementById("rating-error"),
  submitRating: document.getElementById("submit-rating-button"),
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
  state.view = name;
  Object.entries(el.views).forEach(([key, node]) => node.classList.toggle("hidden", key !== name));
  el.signOut.classList.toggle("hidden", !state.token);
  window.scrollTo({ top: 0, behavior: "instant" });
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
  state.ratingPanelId = null;
  state.pendingActions.clear();
  sessionStorage.removeItem(storageKeys.token);
  sessionStorage.removeItem(storageKeys.participant);
  el.access.value = "";
  el.panelHost.replaceChildren();
  showView("login");
}

function handleAuthenticatedError(error) {
  if (error.status === 401) {
    signOut();
    showToast("Your sign-in expired. Please enter your codes again.", true);
    return;
  }
  showToast(error.message, true);
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function profileNumber(profile) {
  return String(Number(profile.display_number) || "").padStart(2, "0");
}

function currentPanel() {
  return state.study?.panels.find(panel => panel.id === state.study.current_panel_id) || null;
}

function sessionPosition(panel) {
  return Number(panel?.display_order ?? state.study?.completed_sessions ?? 0) + 1;
}

function fillList(node, values) {
  node.replaceChildren();
  for (const value of values || []) {
    const item = document.createElement("li");
    item.textContent = value;
    node.appendChild(item);
  }
}

function renderProfiles() {
  el.profileGrid.replaceChildren();
  for (const profile of state.profiles) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "patient-card";
    const completed = profile.study?.completed_sessions || 0;
    const total = profile.study?.total_sessions || 6;
    const statusText = profile.study?.status === "finished"
      ? "Completed"
      : profile.study
        ? `${completed} of ${total} sessions rated`
        : "Not started";
    const actionText = profile.study ? "View profile" : "Open profile";
    button.innerHTML = `
      <span class="patient-number">${profileNumber(profile)}</span>
      <h2>${escapeHtml(profile.display_name)}</h2>
      <span class="condition">${escapeHtml(profile.condition)}</span>
      <p>${escapeHtml(profile.short_description)}</p>
      <span class="study-state">${escapeHtml(statusText)}</span>
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
    state.ratingPanelId = null;
    showPatientProfile();
  } catch (error) {
    handleAuthenticatedError(error);
  }
}

function populatePatientProfile() {
  const profile = state.study?.profile;
  if (!profile) return;
  el.patientName.textContent = profile.display_name;
  el.patientCondition.textContent = profile.condition;
  el.patientSummary.textContent = profile.summary;
  el.patientContext.textContent = profile.current_context;
  el.patientHistory.textContent = profile.relevant_history;
  fillList(el.patientCoping, profile.coping_strategies);
  fillList(el.patientGuidance, profile.role_guidance);
}

function showPatientProfile() {
  clearPoll();
  if (!state.study) return loadProfiles();
  populatePatientProfile();
  const study = state.study;
  const panel = currentPanel();
  el.patientProgress.textContent = `${study.completed_sessions} of ${study.total_sessions} sessions rated`;
  if (study.status === "finished" || !panel) {
    el.patientNextNote.textContent = "All therapist sessions and CTRS evaluations are complete for this patient.";
    el.beginSession.classList.add("hidden");
  } else {
    const position = sessionPosition(panel);
    el.beginSession.classList.remove("hidden");
    el.patientNextNote.textContent = `Next: session ${position} of ${study.total_sessions}`;
    if (panel.status === "rating") {
      el.beginSession.textContent = `Score ${panel.label}`;
    } else if (panel.messages.length || panel.job) {
      el.beginSession.textContent = `Continue with ${panel.label}`;
    } else {
      el.beginSession.textContent = `Begin with ${panel.label}`;
    }
  }
  showView("patient");
}

function openProfileDialog() {
  const profile = state.study?.profile;
  if (!profile) return;
  el.modalName.textContent = `${profile.display_name} - ${profile.condition}`;
  el.modalSummary.textContent = profile.summary;
  el.modalContext.textContent = profile.current_context;
  el.modalHistory.textContent = profile.relevant_history;
  fillList(el.modalCoping, profile.coping_strategies);
  fillList(el.modalGuidance, profile.role_guidance);
  el.profileDialog.showModal();
}

function panelStatus(panel) {
  if (panel.job?.status === "queued") {
    return [`Queued${panel.job.queue_position ? ` #${panel.job.queue_position}` : ""}`, "busy"];
  }
  if (panel.job?.status === "running") return ["Generating...", "busy"];
  if (panel.job?.status === "failed") return ["Response failed", "failed"];
  return [panel.messages.length ? "Ready" : "Not started", ""];
}

function messageSignature(panel) {
  return JSON.stringify({
    ids: panel.messages.map(message => message.id),
    job: panel.job && [panel.job.id, panel.job.status, panel.job.queue_position],
    start: panel.can_start,
    ended: panel.ended_at,
    localPending: state.pendingActions.has(panel.id)
  });
}

function createPanel(panel) {
  const article = document.createElement("article");
  article.className = "chat-panel session-chat";
  article.dataset.panelId = panel.id;
  article.innerHTML = `
    <header class="panel-header">
      <div class="panel-person"><span class="panel-avatar">${escapeHtml(panel.label.slice(-1))}</span><h2>${escapeHtml(panel.label)}</h2></div>
      <span class="panel-status"></span>
    </header>
    <div class="messages" aria-live="polite"></div>
    <form class="composer">
      <textarea rows="1" maxlength="4000" aria-label="Respond as the patient" placeholder="Respond as the patient..."></textarea>
      <button class="send-button" type="submit" aria-label="Send">&#8593;</button>
    </form>`;
  const textarea = article.querySelector("textarea");
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
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
  el.panelHost.replaceChildren(article);
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
  let article = el.panelHost.querySelector(`[data-panel-id="${CSS.escape(panel.id)}"]`);
  if (!article) article = createPanel(panel);
  const [statusText, statusClass] = panelStatus(panel);
  const statusNode = article.querySelector(".panel-status");
  statusNode.textContent = statusText;
  statusNode.className = `panel-status ${statusClass}`.trim();

  const signature = messageSignature(panel);
  const messages = article.querySelector(".messages");
  if (messages.dataset.signature !== signature) {
    const wasAtBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
    messages.replaceChildren();
    for (const message of panel.messages) appendMessage(messages, message);
    if (!panel.messages.length && panel.can_start && !state.pendingActions.has(panel.id)) {
      const empty = document.createElement("div");
      empty.className = "empty-panel";
      empty.innerHTML = `<div><p>Start when you are ready.</p><button class="primary-button" type="button">Start conversation</button></div>`;
      empty.querySelector("button").addEventListener("click", () => startPanel(panel.id));
      messages.appendChild(empty);
    }
    if (["queued", "running"].includes(panel.job?.status) || state.pendingActions.has(panel.id)) {
      const pending = document.createElement("div");
      pending.className = "pending-row";
      pending.textContent = panel.job?.status === "queued"
        ? `Waiting in queue${panel.job.queue_position ? ` - position ${panel.job.queue_position}` : ""}`
        : "Therapist is responding...";
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
    if (wasAtBottom || panel.messages.length <= 2) {
      requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
    }
  }

  const textarea = article.querySelector("textarea");
  const send = article.querySelector(".send-button");
  textarea.disabled = !panel.can_send || state.pendingActions.has(panel.id);
  send.disabled = textarea.disabled || !textarea.value.trim();
}

function showCurrentStage() {
  const panel = currentPanel();
  if (!panel || state.study.status === "finished") return showPatientProfile();
  if (panel.status === "rating") return showRating();
  showSession();
}

function showSession() {
  const panel = currentPanel();
  if (!panel) return showPatientProfile();
  if (panel.status === "rating") return showRating();
  const position = sessionPosition(panel);
  el.sessionProgress.textContent = `Session ${position} of ${state.study.total_sessions}`;
  el.sessionTherapist.textContent = panel.label;
  el.endSession.disabled = !panel.can_end || state.pendingActions.has(panel.id);
  updatePanel(panel);
  showView("session");
  schedulePoll();
}

function hasPendingJob() {
  const panel = currentPanel();
  return Boolean(panel && ["queued", "running"].includes(panel.job?.status));
}

function schedulePoll() {
  clearPoll();
  if (!state.study || !hasPendingJob()) return;
  state.pollTimer = window.setTimeout(refreshStudy, 1400);
}

async function refreshStudy() {
  if (!state.study) return;
  try {
    const payload = await api(`/api/studies/${state.study.id}`);
    state.study = payload.study;
    if (state.view === "session") showSession();
    else if (state.view === "patient") showPatientProfile();
    schedulePoll();
  } catch (error) {
    handleAuthenticatedError(error);
    if (state.study && state.view === "session") {
      state.pollTimer = window.setTimeout(refreshStudy, 3500);
    }
  }
}

async function startPanel(panelId) {
  if (state.pendingActions.has(panelId)) return;
  state.pendingActions.add(panelId);
  showSession();
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
    if (state.view === "session") showSession();
  }
}

async function sendPatientMessage(panelId, rawContent) {
  const content = rawContent.trim();
  if (!content || state.pendingActions.has(panelId)) return;
  const article = el.panelHost.querySelector(`[data-panel-id="${CSS.escape(panelId)}"]`);
  const textarea = article.querySelector("textarea");
  state.pendingActions.add(panelId);
  textarea.value = "";
  showSession();
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
    if (state.view === "session") showSession();
  }
}

async function retryPanel(panelId) {
  if (state.pendingActions.has(panelId)) return;
  state.pendingActions.add(panelId);
  showSession();
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
    if (state.view === "session") showSession();
  }
}

async function endCurrentSession() {
  const panel = currentPanel();
  if (!panel?.can_end || state.pendingActions.has(panel.id)) return;
  if (!window.confirm("End this session and continue to the CTRS evaluation? You will not be able to send more messages.")) return;
  state.pendingActions.add(panel.id);
  el.endSession.disabled = true;
  try {
    const payload = await api(`/api/studies/${state.study.id}/panels/${panel.id}/end`, {
      method: "POST",
      body: JSON.stringify({ client_request_id: requestId() })
    });
    state.study = payload.study;
    showRating();
  } catch (error) {
    handleAuthenticatedError(error);
  } finally {
    state.pendingActions.delete(panel.id);
  }
}

async function leaveCurrentSession(nextView = showPatientProfile) {
  const panel = currentPanel();
  if (!panel || panel.ended_at) {
    nextView();
    return;
  }
  clearPoll();
  try {
    const payload = await api(`/api/studies/${state.study.id}/panels/${panel.id}/leave`, {
      method: "POST",
      body: JSON.stringify({ client_request_id: requestId() })
    });
    state.study = payload.study;
    nextView();
  } catch (error) {
    handleAuthenticatedError(error);
    if (state.token && state.study) showSession();
  }
}

function buildRatingItems() {
  el.ratingItems.replaceChildren();
  let activePart = "";
  for (const item of ctrsRubric) {
    if (item.part !== activePart) {
      activePart = item.part;
      const part = document.createElement("p");
      part.className = "ctrs-part";
      part.textContent = activePart;
      el.ratingItems.appendChild(part);
    }
    const card = document.createElement("section");
    card.className = "ctrs-item";
    card.dataset.key = item.key;

    const heading = document.createElement("div");
    heading.className = "ctrs-item-heading";
    const number = document.createElement("span");
    number.textContent = String(item.number).padStart(2, "0");
    const title = document.createElement("h3");
    title.textContent = item.label;
    heading.append(number, title);
    card.appendChild(heading);

    if (item.note) {
      const note = document.createElement("p");
      note.className = "ctrs-note";
      note.textContent = item.note;
      card.appendChild(note);
    }

    const scores = document.createElement("div");
    scores.className = "score-grid";
    for (const score of ctrsScale) {
      const label = document.createElement("label");
      label.className = "score-option";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = item.key;
      input.value = String(score.value);
      input.required = true;
      input.addEventListener("change", () => {
        scores.querySelectorAll(".score-option").forEach(node => node.classList.toggle("selected", node === label));
        el.ratingError.classList.add("hidden");
      });
      const value = document.createElement("strong");
      value.textContent = score.value;
      const text = document.createElement("small");
      text.textContent = score.label;
      label.append(input, value, text);
      scores.appendChild(label);
    }
    card.appendChild(scores);

    const details = document.createElement("details");
    details.className = "anchor-details";
    const summary = document.createElement("summary");
    summary.textContent = "View anchors";
    details.appendChild(summary);
    const anchors = document.createElement("div");
    anchors.className = "anchor-grid";
    for (const anchorValue of [0, 2, 4, 6]) {
      const anchor = document.createElement("div");
      const badge = document.createElement("strong");
      badge.textContent = String(anchorValue);
      const text = document.createElement("p");
      text.textContent = item.anchors[anchorValue];
      anchor.append(badge, text);
      anchors.appendChild(anchor);
    }
    details.appendChild(anchors);
    card.appendChild(details);
    el.ratingItems.appendChild(card);
  }
}

function renderRatingTranscript(panel) {
  el.ratingTranscript.replaceChildren();
  for (const message of panel.messages) appendMessage(el.ratingTranscript, message);
}

function showRating() {
  clearPoll();
  const panel = currentPanel();
  if (!panel || panel.status !== "rating") return showPatientProfile();
  const position = sessionPosition(panel);
  el.ratingProgress.textContent = `Session ${position} of ${state.study.total_sessions} - CTRS evaluation`;
  el.ratingTitle.textContent = `Score ${panel.label}`;
  const automaticEndMessages = {
    therapist_farewell: "The session ended after the therapist said goodbye.",
    patient_farewell: "The session ended after your farewell.",
    max_turns: `The session ended automatically after ${state.study.max_session_turns || 50} dialogue turns.`
  };
  const automaticEndMessage = automaticEndMessages[panel.termination_reason] || "";
  el.automaticEndNote.textContent = automaticEndMessage;
  el.automaticEndNote.classList.toggle("hidden", !automaticEndMessage);
  renderRatingTranscript(panel);
  if (state.ratingPanelId !== panel.id) {
    state.ratingPanelId = panel.id;
    el.ratingForm.reset();
    el.ratingItems.querySelectorAll(".score-option").forEach(node => node.classList.remove("selected"));
    el.ratingError.classList.add("hidden");
  }
  showView("rating");
}

async function submitRating(event) {
  event.preventDefault();
  const panel = currentPanel();
  if (!panel || panel.status !== "rating") return;
  const scores = {};
  for (const item of ctrsRubric) {
    const selected = el.ratingForm.querySelector(`input[name="${CSS.escape(item.key)}"]:checked`);
    if (!selected) {
      el.ratingError.textContent = "Please select a score for all 11 CTRS items.";
      el.ratingError.classList.remove("hidden");
      el.ratingForm.querySelector(`[data-key="${CSS.escape(item.key)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    scores[item.key] = Number(selected.value);
  }
  el.ratingError.classList.add("hidden");
  el.submitRating.disabled = true;
  try {
    const payload = await api(`/api/studies/${state.study.id}/panels/${panel.id}/rating`, {
      method: "POST",
      body: JSON.stringify({ scores, comments: el.ratingComments.value })
    });
    state.study = payload.study;
    state.ratingPanelId = null;
    showToast(state.study.status === "finished" ? "All six evaluations are complete." : "CTRS evaluation saved.");
    showPatientProfile();
  } catch (error) {
    handleAuthenticatedError(error);
  } finally {
    el.submitRating.disabled = false;
  }
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

el.signOut.addEventListener("click", () => {
  if (state.view === "session") leaveCurrentSession(signOut);
  else signOut();
});
el.patientBack.addEventListener("click", loadProfiles);
el.sessionBack.addEventListener("click", () => leaveCurrentSession(showPatientProfile));
el.home.addEventListener("click", event => {
  event.preventDefault();
  if (!state.token) return;
  if (state.view === "session") leaveCurrentSession(loadProfiles);
  else loadProfiles();
});
el.beginSession.addEventListener("click", showCurrentStage);
el.profileButton.addEventListener("click", openProfileDialog);
el.ratingProfileButton.addEventListener("click", openProfileDialog);
el.closeProfile.addEventListener("click", () => el.profileDialog.close());
el.profileDialog.addEventListener("click", event => {
  if (event.target === el.profileDialog) el.profileDialog.close();
});
el.endSession.addEventListener("click", endCurrentSession);
el.ratingForm.addEventListener("submit", submitRating);

window.addEventListener("pagehide", () => {
  const panel = currentPanel();
  if (!state.token || state.view !== "session" || !panel || panel.ended_at) return;
  const headers = { "Content-Type": "application/json", Authorization: `Bearer ${state.token}` };
  if (usesNgrok) headers["ngrok-skip-browser-warning"] = "1";
  fetch(`${apiBase}/api/studies/${state.study.id}/panels/${panel.id}/leave`, {
    method: "POST",
    headers,
    body: JSON.stringify({ client_request_id: requestId() }),
    keepalive: true
  }).catch(() => {});
});

async function boot() {
  buildRatingItems();
  checkHealth();
  if (state.token) {
    el.participant.value = state.participant;
    await loadProfiles();
  } else {
    showView("login");
  }
}

boot();
