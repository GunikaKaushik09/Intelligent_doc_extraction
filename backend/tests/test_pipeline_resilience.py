import io
from unittest.mock import patch

import fitz

from app.core.exceptions import ExtractionError
from app.schemas.document import DocumentType, ProcessingStatus
from app.services import pipeline


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_llm_failure_returns_graceful_failed_extraction_result():
    """
    A failing LLM/OCR call (bad API key, rate limit, network error, ...)
    must never crash the request -- it should come back as a normal,
    well-formed ProcessingResult with FAILED_EXTRACTION status and a
    human-readable error_detail, so the API contract holds and the
    document still shows up (as failed) on the dashboard.
    """
    content = _pdf_bytes("Total Revenue: 1,000.00\nNet Profit: 200.00\n")

    with patch(
        "app.services.extraction_service.extract_structured_data",
        side_effect=ExtractionError("LLM extraction call failed.", details={"reason": "invalid API key"}),
    ):
        result = pipeline.process_document(
            document_name="pl_test.pdf",
            document_type=DocumentType.PROFIT_AND_LOSS,
            filename="pl_test.pdf",
            content=content,
        )

    assert result.processing_status == ProcessingStatus.FAILED_EXTRACTION
    assert result.error_detail is not None
    assert "invalid API key" in result.error_detail
    assert result.file_validation.is_valid is True  # file itself was fine
    assert result.financial_validations == []


def test_unexpected_exception_during_extraction_is_contained():
    """An unexpected (non-AppError) crash inside the LLM call must also be
    caught and turned into a graceful FAILED_EXTRACTION result, never a
    raw 500 with a leaked stack trace."""
    content = _pdf_bytes("Total Revenue: 1,000.00\n")

    with patch(
        "app.services.extraction_service.extract_structured_data",
        side_effect=RuntimeError("boom"),
    ):
        result = pipeline.process_document(
            document_name="pl_test2.pdf",
            document_type=DocumentType.PROFIT_AND_LOSS,
            filename="pl_test2.pdf",
            content=content,
        )

    assert result.processing_status == ProcessingStatus.FAILED_EXTRACTION
    assert result.error_detail == "LLM extraction failed unexpectedly."
