"""
Pydantic schemas defining the mandatory structured response shape
(spec section 5.2). Every field the evaluator checks for is a first-class,
typed field rather than free-form text.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    PROFIT_AND_LOSS = "profit_and_loss"
    INVOICE = "invoice"


class ProcessingStatus(str, Enum):
    SUCCESS = "SUCCESS"                       # extraction + validation completed
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"       # extracted, but some fields missing/unreadable
    FAILED_VALIDATION = "FAILED_VALIDATION"   # file failed input-control validation
    FAILED_EXTRACTION = "FAILED_EXTRACTION"   # passed input-control but OCR/LLM extraction failed


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# --------------------------------------------------------------------------
# File / input validation
# --------------------------------------------------------------------------
class FileValidationResult(BaseModel):
    is_valid: bool
    file_type: str | None = None
    page_count: int | None = None
    file_size_bytes: int | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Extraction (evidence-grounded key/value + tables)
# --------------------------------------------------------------------------
class Evidence(BaseModel):
    page_number: int | None = None
    source_text: str | None = None


class ExtractedField(BaseModel):
    value: Any = None  # null when not present in the source document
    evidence: Evidence | None = None
    confidence: float | None = None  # optional, 0-1


class ExtractedTableRow(BaseModel):
    """A generic row of a financial statement / invoice line-item table."""

    cells: dict[str, Any]
    evidence: Evidence | None = None


class ExtractedTable(BaseModel):
    name: str
    columns: list[str]
    rows: list[ExtractedTableRow] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    tables: list[ExtractedTable] = Field(default_factory=list)
    raw_ocr_text_excerpt: str | None = None  # first ~1000 chars, for traceability


# --------------------------------------------------------------------------
# Financial validation
# --------------------------------------------------------------------------
class FinancialValidationCheck(BaseModel):
    check_name: str
    formula: str
    input_values: dict[str, Any]
    calculated_value: float | None = None
    reported_value: float | None = None
    variance: float | None = None
    variance_pct: float | None = None
    tolerance_pct: float | None = None
    status: ValidationStatus
    reason: str | None = None  # populated mainly for NOT_APPLICABLE / FAIL


# --------------------------------------------------------------------------
# Processing metadata
# --------------------------------------------------------------------------
class ProcessingMetadata(BaseModel):
    request_id: str
    ocr_engine: str
    llm_model: str | None = None
    processing_time_ms: int
    processed_at: datetime
    page_count: int


class ProcessingResult(BaseModel):
    """The full mandatory structured response (spec 5.2)."""

    document_name: str
    document_type: DocumentType
    file_validation: FileValidationResult
    # Populated only when processing_status is FAILED_EXTRACTION -- a short,
    # user-safe description of what went wrong (e.g. an OCR engine failure or
    # an LLM API error), so the failure is diagnosable from the dashboard
    # and the stored record, not just server logs.
    error_detail: str | None = None
    extracted_data: ExtractionResult
    financial_validations: list[FinancialValidationCheck] = Field(default_factory=list)
    processing_status: ProcessingStatus
    processing_metadata: ProcessingMetadata


class ProcessedDocumentSummary(BaseModel):
    """Row shape for the dashboard / list endpoint."""

    id: str
    document_name: str
    document_type: DocumentType
    status: ProcessingStatus
    created_at: datetime


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str
