# Document Intelligence Platform

Intelligent extraction, financial validation and a REST API + dashboard for
Balance Sheets, Cash Flow Statements, Profit & Loss Statements and Invoices
(PDF / JPG / PNG, up to 3 pages). Built for the AI Engineer Internship
technical case study.

> **Deployed Application URL:** _fill in after deployment_
> **Backend API URL:** _fill in after deployment_
> **Swagger / OpenAPI URL:** _`<backend-url>`/docs_
> **Public GitHub repository:** _fill in after pushing_

---

## 1. Solution overview and architecture

The system is a hybrid OCR + LLM pipeline:

1. **File validation** (input-control, not classification) rejects unsupported
   formats, empty/corrupted files and documents over 3 pages *before* any
   OCR/AI is attempted.
2. **OCR / text extraction** uses native PDF text extraction (PyMuPDF) where
   available, and falls back to Tesseract OCR for scanned pages or image
   uploads -- each page's text is tagged with its page number.
3. **LLM structuring** (Google Gemini by default) turns the raw, page-tagged text into typed
   fields + tables, with per-field evidence (page number + supporting
   snippet), following a strict "never invent a value" instruction. If no
   API key is configured, a deterministic offline heuristic extractor is
   used instead so the whole app remains runnable for free (see
   **Known limitations** below).
4. **Financial validation** recomputes each applicable formula from the
   actually-extracted fields and reports PASS / FAIL / NOT_APPLICABLE with
   the formula, inputs, calculated value, reported value and variance.
5. **Persistence** stores every processed result (SQLite by default) so the
   dashboard and GET-by-name endpoint can retrieve it later.
6. **Frontend** (static HTML/CSS/JS) lets a user pick a document type,
   upload a file, see it processed, browse the dashboard of everything
   processed, and inspect both a human-readable view and the raw JSON.

See [`docs/architecture.md`](docs/architecture.md) for a diagram and a
walk-through of the request flow and the code's separation of concerns.

## 2. Technology stack and reasoning

| Concern              | Choice                        | Why                                                                 |
|-----------------------|-------------------------------|----------------------------------------------------------------------|
| API framework          | FastAPI                       | Native async, automatic Swagger/OpenAPI docs, Pydantic validation out of the box |
| Native PDF text        | PyMuPDF (`fitz`)               | Fast, accurate embedded-text + page rasterization for scanned pages in one library |
| OCR                    | Tesseract (`pytesseract`)      | Free, open-source, no API key needed, works fully offline           |
| LLM structuring        | Google Gemini API (default) or Anthropic Claude | Gemini has a genuinely free tier (no credit card); Claude supported as a drop-in alternative via `LLM_PROVIDER` |
| Data validation/schema | Pydantic v2                   | Enforces the mandatory structured response shape end-to-end          |
| Database               | SQLAlchemy + SQLite (default)  | Zero-config for the assessment; one env var (`DATABASE_URL`) to move to Postgres/MySQL |
| Frontend                | Plain HTML/CSS/JS              | Spec explicitly does not require a separate JS framework; keeps the deployable surface small |

## 3. Local setup instructions

Prerequisites: Python 3.11+, [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) installed and on `PATH` (`apt-get install tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).

```bash
git clone <this-repo-url>
cd document-intelligence/backend

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY if you have one (optional -- see
# "Known limitations" for what happens without it). Get a free key with no
# credit card required at https://aistudio.google.com

uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** for the dashboard and
**http://localhost:8000/docs** for Swagger UI.

### Running tests

```bash
cd backend
USE_MOCK_LLM=true python3 -m pytest -v
```

## 4. Environment variables

See [`backend/.env.example`](backend/.env.example) for the full list (no
real secrets committed). Key ones:

| Variable                | Purpose                                                             |
|--------------------------|----------------------------------------------------------------------|
| `DATABASE_URL`            | SQLAlchemy connection string (defaults to local SQLite file)         |
| `LLM_PROVIDER`            | `gemini` (default, free tier) or `anthropic`                          |
| `GEMINI_API_KEY`          | Google Gemini API key -- free, no credit card, from aistudio.google.com |
| `GEMINI_MODEL`            | Gemini model name, defaults to `gemini-2.5-flash`                      |
| `ANTHROPIC_API_KEY`       | Claude API key, only used if `LLM_PROVIDER=anthropic` (paid account)   |
| `LLM_MODEL`               | Claude model name, defaults to `claude-sonnet-4-5`                    |
| `USE_MOCK_LLM`            | Force the offline heuristic extractor even if a key is set (useful for tests/demos) |
| `MAX_FILE_SIZE_MB`, `MAX_PAGE_COUNT` | Input-control limits                                       |
| `VALIDATION_TOLERANCE_PCT`| Relative tolerance used by every financial validation check          |
| `CORS_ORIGINS`            | Comma-separated allowed origins                                       |

## 5. Deployment (Render, free tier)

The repo includes a Docker image (`backend/Dockerfile`) and a Render
Blueprint (`render.yaml`) that build it and run `uvicorn` behind Render's
free web-service plan (750 free hours/month, custom TLS domain, cold start
after 15 min idle).

**Option A - one-click Blueprint:**
1. Push this repo to a public GitHub repository.
2. In the [Render dashboard](https://dashboard.render.com), click
   **New -> Blueprint** and select your repo. Render reads `render.yaml`
   and provisions the service automatically.
3. Once created, open the service's **Environment** tab and set
   `GEMINI_API_KEY` to your real key (get one free, no credit card, at
   https://aistudio.google.com -- it's intentionally left out of
   `render.yaml` so it's never committed to git).
4. Wait for the first deploy to finish, then visit the assigned
   `https://<service-name>.onrender.com` URL — that's both the dashboard
   (`/`) and the API (`/api/documents/...`); Swagger is at `/docs`.

**Option B - manual web service (no Blueprint):**
1. **New -> Web Service**, connect your GitHub repo.
2. Runtime: **Docker**. Dockerfile path: `backend/Dockerfile`. Docker
   build context: repo root (`.`) — this matters, since the Dockerfile
   copies both `backend/` and `frontend/` (see the comment at the top of
   the Dockerfile).
3. Plan: **Free**. Health check path: `/health`.
4. Add the environment variables listed in `backend/.env.example` (at
   minimum `GEMINI_API_KEY`; everything else has a sensible default).
5. Create Web Service and wait for the build/deploy to finish.

**Persistence note:** Render's free plan doesn't support attached
Persistent Disks (paid-plan feature). The SQLite file survives normal
spin-down/spin-up (the free plan just sleeps the service after 15 minutes
of inactivity and wakes it on the next request) but is reset on every new
deploy. That's fine for a short evaluation window. For longer-lived
persistence without upgrading, point `DATABASE_URL` at a free Render
Postgres instance instead (**New -> PostgreSQL**, free 90-day instance,
then copy its "Internal Database URL" into `DATABASE_URL` and add
`psycopg2-binary` to `requirements.txt`).

**Alternative free platforms:** any host that runs an arbitrary Docker
image works the same way (Fly.io, Google Cloud Run's free tier, etc.) —
build from the repo root with `backend/Dockerfile` and expose port 8000.

## 6. API request examples

**POST /api/documents/process** (multipart upload)

```bash
curl -X POST "http://localhost:8000/api/documents/process" \
  -F "file=@Consolidated Balance Sheet 2024.pdf;type=application/pdf" \
  -F "document_type=balance_sheet" \
  -F "document_name=Balance Sheet 2024"
```

**GET by document name** (returns the latest processed result)

```bash
curl "http://localhost:8000/api/documents/Balance%20Sheet%202024"
```

**Processed-document list (dashboard)**

```bash
curl "http://localhost:8000/api/documents?latest_only=true"
```

Full request/response schemas are in Swagger at `/docs`.

**Troubleshooting: is my LLM key actually being picked up?**

```bash
curl "http://localhost:8000/health/llm-config"
```

Returns which provider/model a new `/process` request will actually use,
whether a key is configured for it, and where on disk the app expects to
find `.env` (`env_file_expected_at`) and whether it actually found one there
(`env_file_found`) -- without ever returning the key itself. The `.env`
path is resolved relative to the app's own location (`backend/.env`), not
your current working directory, so it's found the same way regardless of
which directory you happen to launch `uvicorn` from. If
`api_key_configured` is `false` when you expected `true`, check
`env_file_found` first -- if that's `false`, the key isn't in
`backend/.env` at all (e.g. it was only ever `export`ed in a shell session
that isn't running the server anymore); if it's `true` but the key still
isn't picked up, double-check the variable name is exactly `GEMINI_API_KEY`
with no extra quotes/spaces, and that you restarted the server after
editing `.env` (it's only read at startup).

## 7. OCR / document parsing service and LLM/model used

- **OCR / parsing:** PyMuPDF (native PDF text) + Tesseract OCR (open-source,
  local, free) for scanned pages and image uploads.
- **LLM:** Google Gemini (`gemini-2.5-flash` by default) for structuring OCR
  text into typed fields/tables with evidence. Chosen because Google's
  Gemini API has a genuinely free tier with no credit card required
  (unlike Anthropic/OpenAI, which require a funded account). Claude is also
  supported as a drop-in alternative -- set `LLM_PROVIDER=anthropic` plus
  `ANTHROPIC_API_KEY` if you have a paid account; the extraction service
  (`extraction_service.py`) dispatches to whichever provider is configured
  behind the same prompt and response contract, so swapping providers
  requires no other code changes.
- **Offline fallback:** when no key is configured for the selected provider
  (or `USE_MOCK_LLM=true`), a local regex/keyword-based extractor
  (`extraction_service._offline_extract`) is used instead, so the entire
  pipeline is runnable and testable with zero paid services.

## 8. Confidence scoring

Not implemented. Confidence is explicitly optional per the spec; the schema
supports a `confidence` field per extracted value (nullable) so it can be
added later (e.g. derived from OCR engine confidence scores or LLM
self-reported certainty) without changing the API contract.

## 9. Financial validation rules and tolerance

All checks use a relative tolerance of **1% (`VALIDATION_TOLERANCE_PCT=0.01`)**
to absorb rounding differences, and return `NOT_APPLICABLE` (never an
assumed value) when a required input field was not extracted:

| Document type   | Checks                                                                 |
|------------------|--------------------------------------------------------------------------|
| Balance Sheet     | Total Assets = Current + Non-Current Assets; Total Liabilities = Current + Non-Current Liabilities; Total Assets = Total Liabilities + Total Equity |
| Cash Flow         | Net Change in Cash = Operating + Investing + Financing; Ending Cash = Beginning Cash + Net Change |
| Profit & Loss     | Gross Profit = Revenue − COGS; Operating Income = Gross Profit − Opex; Net Profit = Operating Income − Tax |
| Invoice           | Total = Subtotal + Tax (+ Shipping − Discount); Balance Due = Total − Amount Paid |

See `backend/app/services/financial_validation.py` for the implementation and
`backend/tests/test_financial_validation.py` for PASS/FAIL/NOT_APPLICABLE test
cases.

## 10. Database / persistence approach

SQLAlchemy ORM over SQLite by default (`backend/app/db/models.py`,
`ProcessedDocument` table: id, document_name, document_type,
original_filename, content_type, status, full result JSON, created_at).
Re-processing a document name inserts a new row rather than overwriting;
`GET /api/documents/{name}` returns the row with the latest `created_at` for
that name (prior versions are retained but not exposed via a dedicated
endpoint). Switching to Postgres/MySQL for production is a one-line
`DATABASE_URL` change plus adding the relevant driver to `requirements.txt`.

## 11. Testing

`backend/tests/` (18 tests, `pytest`):
- `test_file_validation.py` -- valid/invalid PDFs, page-limit, corrupted PDF,
  empty file, unsupported extension, valid PNG.
- `test_financial_validation.py` -- PASS, FAIL (with variance) and
  NOT_APPLICABLE (missing fields) scenarios across all four document types.
- `test_api_flow.py` -- full POST → GET-by-name → list flow, an unsupported
  file rejected gracefully with `FAILED_VALIDATION`, and a 404 for an unknown
  document name.
- `test_pipeline_resilience.py` -- an OCR/LLM call failure (bad API key,
  network error, rate limit, ...) or an unexpected engine crash must never
  crash the request: it comes back as a normal `FAILED_EXTRACTION` result
  with a human-readable `error_detail`, and is still persisted so it shows
  up (as failed) on the dashboard rather than silently disappearing.

`sample_outputs/` contains real, unedited JSON captured from live runs of
this exact codebase against the provided dataset (balance sheet, P&L, cash
flow PDFs and an invoice JPG, plus a rejected unsupported file), and two
clearly-labeled illustrative examples showing a full PASS and an explicit
FAIL validation outcome for reference.

## 12. Known limitations

- **Offline extractor coverage.** Without a `GEMINI_API_KEY` (or
  `ANTHROPIC_API_KEY` if using the Claude provider), the fallback extractor
  is a generic regex/keyword matcher and will miss fields on documents
  whose labels don't closely match its patterns (this was visible when
  testing against the provided bank-style consolidated statements). The
  LLM path is designed to handle arbitrary real-world layouts correctly;
  the offline path exists only so the project runs for free without any
  API access at all.
- **Gemini free-tier rate limits.** Google's free tier (as used by default)
  caps requests per minute/day; heavy concurrent testing may hit `429`s,
  which surface as `EXTRACTION_ERROR` responses rather than crashing the app.
- **No document-type auto-classification** -- by design (spec 2/3): the type
  is supplied by the caller/frontend as metadata.
- **No OCR confidence scores** surfaced from Tesseract; only pass/fail-style
  extraction completeness is reported today.
- **Single-tenant, no auth** -- the API has no authentication layer; anyone
  with the URL can upload/view documents. Fine for this assessment, not for
  production.
- **Synchronous processing only** -- large batches of pages/documents are
  processed inline in the request; no background queue.
- **SQLite** is not suited to concurrent-write production workloads.

## 13. What I'd change for production

- Swap the offline heuristic extractor entirely for the LLM path (always
  require an API key), and add automatic retries/backoff for transient LLM
  failures.
- Add authentication/authorization (API keys or OAuth) and per-tenant data
  isolation.
- Move processing to a background job queue (e.g. Celery/RQ) with the POST
  endpoint returning a job id immediately and a status-polling endpoint,
  rather than processing synchronously in the request.
- Move to Postgres, add Alembic migrations, and add object storage (S3-
  compatible) for the original uploaded files instead of processing them
  in-memory only.
- Add structured (JSON) log shipping to an observability platform, request
  tracing, and metrics/alerting on extraction/validation failure rates.
- Add real confidence scoring (e.g. from OCR engine confidences and/or LLM
  self-assessment) and surface it in the dashboard.

## 14. AI coding assistants used

This solution was built with the assistance of Claude (Anthropic), used for:
scaffolding the FastAPI project structure and separation of concerns,
drafting the financial-validation rules engine, writing the OCR/extraction
service code, the prompt templates for LLM-based structuring, the pytest
test suite, and this README. All code was reviewed and smoke-tested against
the provided dataset before being included in the submission.

## 15. Repository structure

```
document-intelligence/
  backend/
    app/
      main.py                    # FastAPI app, CORS, request-id middleware, static mount
      core/                      # config, logging, exception hierarchy
      api/routes/                # documents.py, health.py -- thin HTTP layer
      schemas/document.py        # Pydantic models = the API contract
      services/                  # file_validation, ocr_service, extraction_service,
                                  # financial_validation, storage_service, pipeline
      db/                        # SQLAlchemy engine/session + ORM models
      utils/prompts.py           # per-document-type LLM prompt templates
    tests/                       # pytest suite
    requirements.txt
    .env.example
    Dockerfile
  frontend/
    index.html / style.css / app.js   # dashboard + upload UI
  sample_outputs/                     # real + illustrative JSON examples
  docs/architecture.md                # diagram + request-flow walkthrough
  README.md
```
