# Architecture

```mermaid
flowchart TD
    U["Browser: HTML/CSS/JS Dashboard"] -->|"multipart/form-data\nPOST /api/documents/process"| API[FastAPI App]
    U -->|"GET /api/documents\nGET /api/documents/{name}"| API

    subgraph API["FastAPI Backend"]
        direction TB
        R[API Routes\n(documents.py, health.py)] --> P[Pipeline Orchestrator\n(pipeline.py)]
        P --> FV[File Validation\n(file_validation.py)]
        P --> OCR[OCR / Text Extraction\n(ocr_service.py)\nPyMuPDF native text\n+ Tesseract OCR fallback]
        P --> EX[LLM Extraction\n(extraction_service.py)\nGemini/Claude API structuring\nor offline heuristic fallback]
        P --> FVAL[Financial Validation\n(financial_validation.py)\nPASS / FAIL / NOT_APPLICABLE]
        R --> ST[Storage Service\n(storage_service.py)]
    end

    EX -->|prompt + OCR text| LLM[(Google Gemini API\nor Anthropic Claude API)]
    ST --> DB[(SQLite / Postgres\nProcessedDocument table)]
    API -->|Swagger / OpenAPI| DOCS["/docs"]

    style API fill:#eef4f2,stroke:#1f6f63
```

## Request flow (POST /api/documents/process)

1. **File Validation** -- reject empty/corrupted/unsupported/oversized/too-many-page files *before* any OCR or AI call, per spec 4.1. Returns immediately with `FAILED_VALIDATION` on failure; no downstream processing occurs.
2. **OCR / Text Extraction** -- native PDFs are read directly with PyMuPDF; pages with too little embedded text (a signal of a scanned page) are rasterized and OCR'd with Tesseract. JPG/PNG files always go through Tesseract. Every page of text is tagged with its page number for evidence/grounding.
3. **LLM Extraction** -- the page-tagged OCR text plus a document-type-specific prompt is sent to Google Gemini (default, free tier) or Anthropic Claude, which returns a strict JSON object of fields (with per-field evidence: page number + supporting snippet) and tables. If no API key is configured for the selected `LLM_PROVIDER`, a deterministic offline regex-based extractor is used instead so the app remains fully runnable/testable for free.
4. **Financial Validation** -- a small rules engine (one module per document type) recomputes each check's formula from the *actually extracted* fields. Any check whose inputs are missing returns `NOT_APPLICABLE`; only checks whose inputs are all present are evaluated as `PASS`/`FAIL` against a configurable tolerance.
5. **Persistence** -- the full structured result is written to the database (SQLite by default; swap `DATABASE_URL` for any SQLAlchemy-supported store e.g. Postgres/MySQL). Re-processing the same `document_name` inserts a new row; `GET /api/documents/{name}` returns the most recent one.
6. **Response** -- the same structured JSON object is returned to the caller and is what the dashboard renders.

## Why this separation of concerns

Each concern (validation / OCR / extraction / financial rules / persistence / API) lives in its own module with a narrow, testable interface, per spec section 8/9:

```
backend/app/
  core/        # config, logging, exception hierarchy - shared plumbing
  api/routes/  # thin HTTP layer only - no business logic
  services/    # file_validation, ocr_service, extraction_service,
               # financial_validation, storage_service, pipeline (orchestrator)
  schemas/     # Pydantic models = the API contract
  db/          # SQLAlchemy engine/session + ORM models
  utils/       # prompt templates
```

This means, for example, swapping Tesseract for Google Cloud Vision only touches `ocr_service.py`; swapping SQLite for Postgres only touches `DATABASE_URL` and `db/database.py`; adding a fifth document type only touches `prompts.py` + `financial_validation.py` + the `DocumentType` enum.
