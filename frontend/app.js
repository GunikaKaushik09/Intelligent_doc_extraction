const API_BASE = "/api/documents";

const form = document.getElementById("upload-form");
const statusBox = document.getElementById("upload-status");
const dashboardBody = document.getElementById("dashboard-body");
const refreshBtn = document.getElementById("refresh-btn");
const modal = document.getElementById("detail-modal");
const detailTitle = document.getElementById("detail-title");
const detailBody = document.getElementById("detail-body");
const rawJsonView = document.getElementById("raw-json-view");
const toggleJsonBtn = document.getElementById("toggle-json-btn");
const closeModalBtn = document.getElementById("close-modal-btn");

let currentResult = null;

function showStatus(message, isError = false) {
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function statusPillClass(status) {
  if (status === "SUCCESS") return "success";
  if (status === "PARTIAL_SUCCESS") return "partial";
  return "failed";
}

async function loadDashboard() {
  try {
    const resp = await fetch(`${API_BASE}?latest_only=true`);
    if (!resp.ok) throw new Error("Could not load the document list.");
    const docs = await resp.json();

    if (!docs.length) {
      dashboardBody.innerHTML = `<tr><td colspan="4" class="empty-row">No documents processed yet.</td></tr>`;
      return;
    }

    dashboardBody.innerHTML = docs
      .map(
        (doc) => `
      <tr data-name="${encodeURIComponent(doc.document_name)}">
        <td>${doc.document_name}</td>
        <td>${doc.document_type.replace(/_/g, " ")}</td>
        <td><span class="pill ${statusPillClass(doc.status)}">${doc.status}</span></td>
        <td>${new Date(doc.created_at).toLocaleString()}</td>
      </tr>`
      )
      .join("");

    dashboardBody.querySelectorAll("tr[data-name]").forEach((row) => {
      row.addEventListener("click", () => openDetail(decodeURIComponent(row.dataset.name)));
    });
  } catch (err) {
    dashboardBody.innerHTML = `<tr><td colspan="4" class="empty-row">${err.message}</td></tr>`;
  }
}

function fieldRow(key, field) {
  const missing = field.value === null || field.value === undefined;
  const evidence = field.evidence
    ? `<span class="evidence">${field.evidence.page_number ? `p.${field.evidence.page_number} — ` : ""}${
        field.evidence.source_text || ""
      }</span>`
    : "";
  return `<div class="kv-item">
      <span class="key">${key.replace(/_/g, " ")}</span>
      <span class="value ${missing ? "missing" : ""}">${missing ? "Not found in document" : field.value}</span>
      ${evidence}
    </div>`;
}

function renderTable(table) {
  const cols = table.columns.length ? table.columns : Object.keys(table.rows[0]?.cells || {});
  const rows = table.rows
    .map((r) => `<tr>${cols.map((c) => `<td>${r.cells[c] ?? "—"}</td>`).join("")}</tr>`)
    .join("");
  return `<h4>${table.name}</h4>
    <table class="data-table">
      <thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>${rows || `<tr><td colspan="${cols.length}">No rows extracted.</td></tr>`}</tbody>
    </table>`;
}

function renderCheck(check) {
  return `<div class="check-row ${check.status}">
      <div class="check-name">${check.check_name}</div>
      <div class="check-meta">${check.formula}</div>
      <div class="check-meta">
        calculated: ${check.calculated_value ?? "—"} &nbsp;|&nbsp;
        reported: ${check.reported_value ?? "—"} &nbsp;|&nbsp;
        variance: ${check.variance ?? "—"} &nbsp;|&nbsp;
        status: <strong>${check.status}</strong>
        ${check.reason ? `<br/>${check.reason}` : ""}
      </div>
    </div>`;
}

function renderDetail(result) {
  const fields = result.extracted_data.fields || {};
  const tables = result.extracted_data.tables || [];
  const checks = result.financial_validations || [];

  const fieldsHtml = Object.keys(fields).length
    ? `<div class="kv-grid">${Object.entries(fields).map(([k, v]) => fieldRow(k, v)).join("")}</div>`
    : "<p>No key-value fields were extracted.</p>";

  const tablesHtml = tables.length ? tables.map(renderTable).join("") : "<p>No tables were extracted.</p>";

  const checksHtml = checks.length
    ? checks.map(renderCheck).join("")
    : "<p>No financial validation checks apply to this document type.</p>";

  const fv = result.file_validation;
  const meta = result.processing_metadata;

  detailBody.innerHTML = `
    <div class="detail-section">
      <h3>Overview</h3>
      <div class="kv-grid">
        <div class="kv-item"><span class="key">status</span><span class="value">${result.processing_status}</span></div>
        <div class="kv-item"><span class="key">file type</span><span class="value">${fv.file_type ?? "—"}</span></div>
        <div class="kv-item"><span class="key">page count</span><span class="value">${fv.page_count ?? "—"}</span></div>
        <div class="kv-item"><span class="key">ocr engine</span><span class="value">${meta.ocr_engine}</span></div>
        <div class="kv-item"><span class="key">model</span><span class="value">${meta.llm_model ?? "—"}</span></div>
        <div class="kv-item"><span class="key">processed at</span><span class="value">${new Date(meta.processed_at).toLocaleString()}</span></div>
      </div>
      ${fv.errors && fv.errors.length ? `<p style="color:var(--fail)">${fv.errors.join("; ")}</p>` : ""}
      ${result.error_detail ? `<p style="color:var(--fail)"><strong>Processing error:</strong> ${result.error_detail}</p>` : ""}
    </div>

    <div class="detail-section">
      <h3>Extracted Fields</h3>
      ${fieldsHtml}
    </div>

    <div class="detail-section">
      <h3>Extracted Tables</h3>
      ${tablesHtml}
    </div>

    <div class="detail-section">
      <h3>Financial Validation</h3>
      ${checksHtml}
    </div>
  `;
}

async function openDetail(documentName) {
  try {
    const resp = await fetch(`${API_BASE}/${encodeURIComponent(documentName)}`);
    if (!resp.ok) throw new Error("Could not load this document's result.");
    currentResult = await resp.json();
    detailTitle.textContent = documentName;
    renderDetail(currentResult);
    rawJsonView.textContent = JSON.stringify(currentResult, null, 2);
    rawJsonView.hidden = true;
    detailBody.hidden = false;
    toggleJsonBtn.textContent = "View Raw JSON";
    modal.hidden = false;
  } catch (err) {
    alert(err.message);
  }
}

toggleJsonBtn.addEventListener("click", () => {
  const showingJson = !rawJsonView.hidden;
  rawJsonView.hidden = showingJson;
  detailBody.hidden = !showingJson;
  toggleJsonBtn.textContent = showingJson ? "View Raw JSON" : "View Extracted Data";
});

closeModalBtn.addEventListener("click", () => (modal.hidden = true));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.hidden = true;
});

refreshBtn.addEventListener("click", loadDashboard);

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const processBtn = document.getElementById("process-btn");
  const fileInput = document.getElementById("file");

  if (!fileInput.files.length) {
    showStatus("Please choose a file to upload.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("document_type", document.getElementById("document_type").value);
  const docName = document.getElementById("document_name").value.trim();
  if (docName) formData.append("document_name", docName);

  processBtn.disabled = true;
  processBtn.textContent = "Processing…";
  showStatus("Uploading and processing document — this can take a few seconds…");

  try {
    const resp = await fetch(`${API_BASE}/process`, { method: "POST", body: formData });
    const body = await resp.json();
    if (!resp.ok) {
      const reason = body.details && body.details.reason ? ` (${body.details.reason})` : "";
      throw new Error((body.message || "Processing failed.") + reason);
    }

    showStatus(`Processed "${body.document_name}" — status: ${body.processing_status}.`);
    form.reset();
    await loadDashboard();
    openDetail(body.document_name);
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "Process Document";
  }
});

loadDashboard();
