"""
Input-control layer: validates an uploaded file *before* any OCR/AI
extraction is attempted (spec 4.1). This is deliberately separate from
document-type classification -- we never try to guess the document type
here, only whether the bytes are a well-formed, in-scope, size/page-limited
file.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.schemas.document import FileValidationResult

logger = logging.getLogger("document_intelligence")
settings = get_settings()


def _extension_ok(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return ext if ext in settings.ALLOWED_EXTENSIONS else None


def _validate_pdf(content: bytes, errors: list[str]) -> int | None:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:  # corrupted / unreadable PDF
        errors.append(f"Corrupted or unreadable PDF: {exc}")
        return None

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            errors.append("PDF is password-protected and cannot be read.")
            return None
        page_count = doc.page_count
        if page_count == 0:
            errors.append("PDF contains no pages.")
            return None
        if page_count > settings.MAX_PAGE_COUNT:
            errors.append(
                f"PDF has {page_count} pages, which exceeds the "
                f"{settings.MAX_PAGE_COUNT}-page limit."
            )
        return page_count
    finally:
        doc.close()


def _validate_image(content: bytes, errors: list[str]) -> int | None:
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # raises if truncated/corrupted
        return 1
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        errors.append(f"Corrupted or unreadable image: {exc}")
        return None


def validate_file(filename: str, content: bytes) -> FileValidationResult:
    errors: list[str] = []

    if not content:
        return FileValidationResult(
            is_valid=False,
            file_type=None,
            page_count=None,
            file_size_bytes=0,
            errors=["Uploaded file is empty."],
        )

    size_bytes = len(content)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        errors.append(
            f"File size {size_bytes} bytes exceeds the {settings.MAX_FILE_SIZE_MB}MB limit."
        )

    ext = _extension_ok(filename)
    if ext is None:
        errors.append(
            f"Unsupported file type '{Path(filename).suffix}'. "
            f"Only PDF, JPG and PNG are supported."
        )
        return FileValidationResult(
            is_valid=False,
            file_type=Path(filename).suffix.lstrip(".").lower() or None,
            page_count=None,
            file_size_bytes=size_bytes,
            errors=errors,
        )

    page_count: int | None
    if ext == ".pdf":
        page_count = _validate_pdf(content, errors)
        file_type = "pdf"
    else:
        page_count = _validate_image(content, errors)
        file_type = "jpg" if ext in (".jpg", ".jpeg") else "png"

    is_valid = len(errors) == 0 and page_count is not None
    result = FileValidationResult(
        is_valid=is_valid,
        file_type=file_type,
        page_count=page_count,
        file_size_bytes=size_bytes,
        errors=errors,
    )
    logger.info(
        "File validation completed",
        extra={"request_id": "-"},
    )
    return result
