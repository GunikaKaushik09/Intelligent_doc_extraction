"""
Orchestrates the full per-document pipeline:
  1. file validation (input-control)
  2. OCR / text extraction
  3. LLM-based structuring into fields/tables
  4. financial validation
  5. assembly of the mandatory structured JSON response

Kept deliberately thin -- all real logic lives in the single-purpose
services it calls, per the required separation of concerns (spec 8/9).
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.schemas.document import (
    DocumentType,
    ExtractionResult,
    FileValidationResult,
    ProcessingMetadata,
    ProcessingResult,
    ProcessingStatus,
)
from app.services import extraction_service, financial_validation, ocr_service
from app.services.file_validation import validate_file

logger = logging.getLogger("document_intelligence")
settings = get_settings()


def _failed_extraction_result(
    *, document_name, document_type, file_validation, request_id, started, ocr_engine, llm_model, error_detail
) -> ProcessingResult:
    return ProcessingResult(
        document_name=document_name,
        document_type=document_type,
        file_validation=file_validation,
        extracted_data=ExtractionResult(),
        financial_validations=[],
        processing_status=ProcessingStatus.FAILED_EXTRACTION,
        error_detail=error_detail,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            ocr_engine=ocr_engine,
            llm_model=llm_model,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            processed_at=datetime.now(timezone.utc),
            page_count=file_validation.page_count or 0,
        ),
    )


def process_document(
    *, document_name: str, document_type: DocumentType, filename: str, content: bytes, request_id: str | None = None
) -> ProcessingResult:
    request_id = request_id or str(uuid.uuid4())
    started = time.perf_counter()

    file_validation = validate_file(filename, content)

    if not file_validation.is_valid:
        logger.warning(
            "File failed input validation",
            extra={"request_id": request_id},
        )
        return ProcessingResult(
            document_name=document_name,
            document_type=document_type,
            file_validation=file_validation,
            extracted_data=ExtractionResult(),
            financial_validations=[],
            processing_status=ProcessingStatus.FAILED_VALIDATION,
            processing_metadata=ProcessingMetadata(
                request_id=request_id,
                ocr_engine="n/a",
                llm_model=None,
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                processed_at=datetime.now(timezone.utc),
                page_count=file_validation.page_count or 0,
            ),
        )

    # OCR and LLM calls hit external engines/services that can legitimately
    # fail (corrupt render, network error, invalid/missing API key, rate
    # limit, timeout, malformed model response, ...). Any such failure must
    # still produce a valid, persisted, FAILED_EXTRACTION response rather
    # than crashing the request -- "fail gracefully" (spec 4.1/9) applies to
    # OCR/model failures just as much as bad input files.
    try:
        ocr_result = ocr_service.extract_text(file_validation.file_type, content)
        logger.info("OCR/text extraction completed", extra={"request_id": request_id})
    except AppError as exc:
        logger.warning(
            "OCR/text extraction failed: %s | details=%s", exc.message, exc.details,
            extra={"request_id": request_id},
        )
        return _failed_extraction_result(
            document_name=document_name, document_type=document_type, file_validation=file_validation,
            request_id=request_id, started=started, ocr_engine="failed", llm_model=None,
            error_detail=exc.message,
        )
    except Exception as exc:  # unexpected engine crash -- never leak the raw exception
        logger.exception("Unexpected OCR failure", extra={"request_id": request_id})
        return _failed_extraction_result(
            document_name=document_name, document_type=document_type, file_validation=file_validation,
            request_id=request_id, started=started, ocr_engine="failed", llm_model=None,
            error_detail="OCR processing failed unexpectedly.",
        )

    try:
        extraction, model_label = extraction_service.extract_structured_data(document_type.value, ocr_result)
        logger.info("LLM structuring completed", extra={"request_id": request_id})
    except AppError as exc:
        logger.warning(
            "LLM extraction failed: %s | details=%s", exc.message, exc.details,
            extra={"request_id": request_id},
        )
        return _failed_extraction_result(
            document_name=document_name, document_type=document_type, file_validation=file_validation,
            request_id=request_id, started=started, ocr_engine=ocr_result.engine_used, llm_model=None,
            error_detail=exc.message + (f" ({exc.details.get('reason')})" if exc.details.get("reason") else ""),
        )
    except Exception as exc:  # unexpected SDK/parsing crash -- never leak the raw exception
        logger.exception("Unexpected LLM extraction failure", extra={"request_id": request_id})
        return _failed_extraction_result(
            document_name=document_name, document_type=document_type, file_validation=file_validation,
            request_id=request_id, started=started, ocr_engine=ocr_result.engine_used, llm_model=None,
            error_detail="LLM extraction failed unexpectedly.",
        )

    checks = financial_validation.run_financial_validations(document_type, extraction)

    extracted_values = [f.value for f in extraction.fields.values()]
    has_any_value = any(v is not None for v in extracted_values)
    has_missing = any(v is None for v in extracted_values)
    if not has_any_value:
        status = ProcessingStatus.FAILED_EXTRACTION
    elif has_missing:
        status = ProcessingStatus.PARTIAL_SUCCESS
    else:
        status = ProcessingStatus.SUCCESS

    result = ProcessingResult(
        document_name=document_name,
        document_type=document_type,
        file_validation=file_validation,
        extracted_data=extraction,
        financial_validations=checks,
        processing_status=status,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            ocr_engine=ocr_result.engine_used,
            llm_model=model_label,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            processed_at=datetime.now(timezone.utc),
            page_count=file_validation.page_count or 0,
        ),
    )
    return result
