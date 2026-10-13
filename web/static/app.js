"use strict";
// Thin client: everything displayed comes from the backend (/api/claims/{id}); nothing is simulated here.

const $ = (sel) => document.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const list = (items) => (items && items.length ? `<ul class="clean">${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : `<p class="hint">None.</p>`);
const label = (s) => esc(String(s ?? "—").replaceAll("_", " "));

let pollTimer = null;
let startedAt = null;

// ------------------------------------------------------------------ health & demo cases
async function loadHealth() {
  const el = $("#health");
  try {
    const h = await (await fetch("/api/health")).json();
    el.textContent = h.ok ? `Ollama OK · ${h.text_model} · vision ${h.vision_model} · ${h.embedding_model} · coordinator: ${h.coordinator_mode}` : `Backend problem: ${h.ollama}`;
    el.className = "health " + (h.ok ? "ok" : "bad");
  } catch {
    el.textContent = "Backend unreachable";
    el.className = "health bad";
  }
}

let demoCases = [];
async function loadDemoCases() {
  demoCases = await (await fetch("/api/demo-cases")).json();
  const sel = $("#demo-select");
  for (const c of demoCases) sel.insertAdjacentHTML("beforeend", `<option value="${esc(c.id)}">${esc(c.title)}</option>`);
}

// Fills the form AND attaches the demo files as real File objects, so they go through the same upload path.
async function applyDemoCase(id) {
  const c = demoCases.find((x) => x.id === id);
  if (!c) return;
  const form = $("#claim-form");
  form.reset();
  $("#form-errors").hidden = true;
  $("#description").value = c.description;
  for (const [k, v] of Object.entries(c.metadata || {})) if (form.elements[k]) form.elements[k].value = v;
  const attach = async (input, names) => {
    const dt = new DataTransfer();
    for (const n of names) {
      const blob = await (await fetch(`/api/demo-cases/${encodeURIComponent(id)}/files/${encodeURIComponent(n)}`)).blob();
      dt.items.add(new File([blob], n, { type: blob.type }));
    }
    input.files = dt.files;
  };
  await attach($("#images"), c.images || []);
  await attach($("#documents"), c.documents || []);
  showFiles();
}

function showFiles() {
  const names = [...$("#images").files].map((f) => `📷 ${f.name}`).concat([...$("#documents").files].map((f) => `📄 ${f.name}`));
  $("#file-list").innerHTML = names.map((n) => `<span>${esc(n)}</span>`).join("");
}

// ------------------------------------------------------------------ submit & poll
async function submitClaim(ev) {
  ev.preventDefault();
  const errBox = $("#form-errors");
  errBox.hidden = true;
  const btn = $("#submit-btn");
  btn.disabled = true;
  btn.textContent = "Submitting…";
  try {
    const resp = await fetch("/api/claims", { method: "POST", body: new FormData($("#claim-form")) });
    const data = await resp.json();
    if (!resp.ok) {
      const errors = data.errors || (data.detail || []).map((d) => d.msg || JSON.stringify(d));
      errBox.innerHTML = `<b>The claim could not be submitted:</b><ul>${errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>`;
      errBox.hidden = false;
      btn.disabled = false;
      btn.textContent = "Run triage";
      return;
    }
    startRun(data.job_id);
  } catch (e) {
    errBox.innerHTML = `<b>Network error:</b> ${esc(e.message)}`;
    errBox.hidden = false;
    btn.disabled = false;
    btn.textContent = "Run triage";
  }
}

function startRun(jobId) {
  $("#run").hidden = false;
  $("#report").hidden = true;
  $("#trace").innerHTML = "";
  startedAt = Date.now();
  $("#submit-btn").textContent = "Running…";
  clearInterval(pollTimer);
  pollTimer = setInterval(() => poll(jobId), 1500);
  poll(jobId);
  $("#run").scrollIntoView({ behavior: "smooth" });
}

async function poll(jobId) {
  let job;
  try {
    job = await (await fetch(`/api/claims/${jobId}`)).json();
  } catch {
    return; // transient; next tick retries
  }
  renderTrace(job.steps || []);
  const st = $("#run-status");
  const secs = Math.round((Date.now() - startedAt) / 1000);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(pollTimer);
    $("#submit-btn").disabled = false;
    $("#submit-btn").textContent = "Run triage";
    loadRuns();
  }
  if (job.status === "completed") {
    st.className = "status done";
    st.innerHTML = `<span class="dot"></span>Completed · ${job.steps.length} steps · job ${esc(job.job_id)}`;
    renderReport(job.report);
  } else if (job.status === "failed") {
    st.className = "status failed";
    st.innerHTML = `<span class="dot"></span>Failed: ${esc(job.error)}`;
  } else {
    st.className = "status";
    st.innerHTML = `<span class="dot"></span>${esc(job.status)} · ${job.steps.length} steps done · ${secs}s elapsed (local models: a step can take 10–60 s)`;
  }
}

function renderTrace(steps) {
  $("#trace").innerHTML = steps.map((s) => `
    <li class="${s.status === "error" ? "error" : ""}">
      <div class="row1"><span class="n">${s.step}</span><span class="act">${esc(s.action)}</span>
        ${s.target ? `<span class="tgt">${esc(s.target)}</span>` : ""}
        <span class="chip">${label(s.agent)}</span><span class="chip ${esc(s.decided_by)}">decided by: ${label(s.decided_by)}</span>
        <span class="chip">${(s.duration_ms / 1000).toFixed(1)} s</span>${s.status !== "ok" ? `<span class="chip st fail">${esc(s.status)}</span>` : ""}</div>
      <div class="reason"><b>Reason:</b> ${esc(s.reason)}</div>
      <div class="obs">${esc(s.observation)}</div>
    </li>`).join("");
}

// ------------------------------------------------------------------ report
function renderReport(r) {
  const el = $("#report");
  el.hidden = false;
  const c = r.confidence;
  el.innerHTML = `
    <div class="card-head"><h2>3 · Triage report <span class="hint">${esc(r.report_id)} · claim ${esc(r.claim_id)} · ${esc(r.generated_at)}</span></h2>
      <button class="secondary" id="dl">Download JSON</button></div>
    <div class="report-head">
      <div class="prio ${esc(r.priority.level)}"><div class="lvl">${esc(r.priority.level)}</div><div class="lab">${esc(r.priority.label)} · score ${r.priority.score}</div></div>
      <div class="kpis">
        <div class="kpi"><div class="k">Primary damage</div><div class="v">${label(r.primary_damage_type)}</div><div class="hint">${r.secondary_damage_types.map(label).join(", ")}</div></div>
        <div class="kpi"><div class="k">Evidence confidence</div><div class="v">${esc(c.level)}</div><div class="hint">mean score ${c.score} before caps · heuristic, not a probability</div></div>
        <div class="kpi"><div class="k">Narrative source</div><div class="v">${label(r.narrative_source)}</div><div class="hint">run: ${label(r.termination)}</div></div>
      </div>
    </div>
    <div class="review ${r.needs_human_review ? "" : "no"}"><b>${r.needs_human_review ? "Human review recommended" : "No review flags raised"}</b>${r.needs_human_review ? list(r.human_review_reasons) : ""}</div>
    <h3>Executive summary <span class="chip inferred">system-generated</span></h3><p class="summary">${esc(r.executive_summary)}</p>

    <h3>Priority factors (deterministic rubric)</h3>
    <div class="table-wrap"><table><tr><th>Factor</th><th>Points</th><th>Source</th></tr>
      ${r.priority.factors.map((f) => `<tr><td>${esc(f.factor)}</td><td>+${f.points}</td><td>${esc(f.source)}</td></tr>`).join("") || `<tr><td colspan="3">No priority factors present.</td></tr>`}
    </table></div><p class="hint">${esc(r.priority.scale)}</p>

    <h3>Facts and their sources</h3>
    <div class="table-wrap"><table><tr><th>Type</th><th>Statement</th><th>Source</th></tr>
      ${r.facts.map((f) => `<tr><td><span class="chip ${esc(f.category)}">${esc(f.category)}</span></td><td>${esc(f.statement)}</td><td>${esc(f.source)}</td></tr>`).join("")}
    </table></div>

    <div class="cols">
      <div><h3>Photo analysis (vision model)</h3>${r.image_findings.length ? r.image_findings.map((i) => `
        <p><b>${esc(i.filename)}</b> — ${i.processed ? `damage visible: ${i.damage_visible ? "yes" : "no"}, severity: ${esc(i.severity)}, kind: ${label(i.image_kind)}` : `<span class="st fail">not processed</span>: ${esc(i.error)}`}</p>
        ${i.processed ? list(i.observations.concat(i.hazards_visible.map((h) => "Hazard: " + h))) : ""}`).join("") : `<p class="hint">No photos submitted.</p>`}</div>
      <div><h3>Documents</h3>${r.document_findings.length ? r.document_findings.map((d) => `
        <p><b>${esc(d.filename)}</b> — ${d.processed ? `${label(d.document_type)}, amounts: ${d.amounts_eur.map((a) => a.toLocaleString("en", { minimumFractionDigits: 2 })).join(", ") || "none"} EUR, dates: ${d.dates.join(", ") || "none"}` : `<span class="st fail">not processed</span>: ${esc(d.error)}`}</p>
        ${d.processed ? list(d.key_facts) : ""}`).join("") : `<p class="hint">No documents submitted.</p>`}</div>
    </div>

    <h3>Similar historical claims (vector search) <span class="chip synthetic">synthetic knowledge base</span></h3>
    <div class="table-wrap"><table><tr><th>Claim</th><th>Similarity</th><th>Type / severity</th><th>Outcome</th><th>Why similar</th></tr>
      ${r.similar_claims.map((h) => `<tr><td><b>${esc(h.claim_id)}</b><div class="hint">${esc(h.summary)}</div></td>
        <td>${h.similarity.toFixed(3)}<div class="bar"><i style="width:${Math.max(0, Math.min(100, (h.similarity - 0.5) * 250))}%"></i></div></td>
        <td>${label(h.damage_type)}<div class="hint">${esc(h.severity)}</div></td>
        <td>${label(h.outcome)}${h.settled_amount_eur ? `<div class="hint">${h.settled_amount_eur.toLocaleString("en")} EUR · ${h.days_to_close} d</div>` : ""}${h.flags.length ? `<div class="hint">flags: ${h.flags.map(label).join(", ")}</div>` : ""}</td>
        <td>${list(h.why_similar)}</td></tr>`).join("") || `<tr><td colspan="5">No historical claim above the similarity threshold.</td></tr>`}
    </table></div>
    <h3>Historical insights</h3>${list(r.historical_insights)}

    <h3>Evidence consistency checks (deterministic)</h3>
    <div class="table-wrap"><table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>
      ${r.consistency_checks.map((k) => `<tr><td>${label(k.check)}</td><td class="st ${esc(k.status)}">${label(k.status)}</td><td>${esc(k.detail)}</td></tr>`).join("")}
    </table></div>

    <h3>Preliminary assessment <span class="chip inferred">inference</span></h3><p>${esc(r.preliminary_assessment)}</p>
    <div class="cols">
      <div><h3>Recommended actions <span class="chip inferred">recommendation</span></h3>${list(r.recommended_actions)}</div>
      <div><h3>Information to request</h3>${list(r.information_to_request)}</div>
    </div>
    <h3>Uncertainties and limitations</h3>${list(r.uncertainties)}
    <h3>Confidence components</h3><p class="hint">${Object.entries(c.components).map(([k, v]) => `${label(k)}: ${v}`).join(" · ")} — ${esc(c.note)}</p>
    <p class="disclaimer">${esc(r.disclaimer)}</p>`;
  $("#dl").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], { type: "application/json" }));
    a.download = `${r.claim_id}_${r.report_id}.json`;
    a.click();
  };
}

// ------------------------------------------------------------------ previous runs
async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  $("#runs-list").innerHTML = runs.map((r) => `<li><a data-id="${esc(r.job_id)}">${esc(r.claim_id || r.job_id)}</a> · ${esc(r.priority || r.status)} · ${esc(r.created_at)}</li>`).join("") || "<li>No runs yet.</li>";
  for (const a of $("#runs-list").querySelectorAll("a")) a.onclick = () => startRun(a.dataset.id);
}

function resetAll() {
  clearInterval(pollTimer);
  $("#claim-form").reset();
  $("#demo-select").value = "";
  $("#file-list").innerHTML = "";
  $("#form-errors").hidden = true;
  $("#run").hidden = true;
  $("#report").hidden = true;
  $("#submit-btn").disabled = false;
  $("#submit-btn").textContent = "Run triage";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("#claim-form").addEventListener("submit", submitClaim);
$("#reset-btn").addEventListener("click", resetAll);
$("#demo-select").addEventListener("change", (e) => applyDemoCase(e.target.value));
$("#images").addEventListener("change", showFiles);
$("#documents").addEventListener("change", showFiles);
loadHealth();
loadDemoCases();
loadRuns();
