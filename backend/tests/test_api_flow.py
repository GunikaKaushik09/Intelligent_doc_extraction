import io
import os

os.environ["USE_MOCK_LLM"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./data/test_document_intelligence.db"

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app

# Using the context-manager form ensures FastAPI's startup event (which
# creates the database tables) actually runs before the tests execute.
client = TestClient(app)
client.__enter__()


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_llm_config_diagnostics_reports_mock_when_no_key():
    # USE_MOCK_LLM=true is set at module import time above, so this always
    # reports the mock path regardless of any real key in the environment.
    resp = client.get("/health/llm-config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["next_request_will_use"] == "offline-document-aware-extractor"
    assert body["use_mock_llm_flag"] is True
    assert body["env_file_expected_at"].endswith("backend/.env")
    assert "env_file_found" in body


def test_process_balance_sheet_and_retrieve():
    content = _pdf_bytes(
        "CONSOLIDATED BALANCE SHEET\n"
        "Total Current Assets: 1,000.00\n"
        "Total Non-Current Assets: 4,000.00\n"
        "Total Assets: 5,000.00\n"
    )
    files = {"file": ("balance_sheet_test.pdf", io.BytesIO(content), "application/pdf")}
    data = {"document_type": "balance_sheet", "document_name": "balance_sheet_test.pdf"}

    resp = client.post("/api/documents/process", files=files, data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_name"] == "balance_sheet_test.pdf"
    assert body["file_validation"]["is_valid"] is True
    assert body["processing_status"] in ("SUCCESS", "PARTIAL_SUCCESS")

    # Retrieve by name -> should return the same/latest result.
    get_resp = client.get("/api/documents/balance_sheet_test.pdf")
    assert get_resp.status_code == 200
    assert get_resp.json()["document_name"] == "balance_sheet_test.pdf"

    # Dashboard list should include it.
    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    names = [d["document_name"] for d in list_resp.json()]
    assert "balance_sheet_test.pdf" in names


def test_unsupported_file_type_rejected_gracefully():
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}
    data = {"document_type": "invoice", "document_name": "notes.txt"}
    resp = client.post("/api/documents/process", files=files, data=data)
    assert resp.status_code == 200  # validation failure is a normal, well-formed response
    body = resp.json()
    assert body["file_validation"]["is_valid"] is False
    assert body["processing_status"] == "FAILED_VALIDATION"


def test_get_unknown_document_returns_404():
    resp = client.get("/api/documents/does-not-exist.pdf")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "DOCUMENT_NOT_FOUND"
