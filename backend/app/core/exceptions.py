"""
Custom exception hierarchy + FastAPI exception handlers.

Goals:
  * Never leak stack traces, internal paths or secrets to the client.
  * Return a consistent error JSON shape across the whole API.
  * Map each failure category to an appropriate HTTP status code.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("document_intelligence")


class AppError(Exception):
    """Base class for all application-raised (expected) errors."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "APP_ERROR"

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class FileValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "FILE_VALIDATION_ERROR"


class UnsupportedFileTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    error_code = "UNSUPPORTED_FILE_TYPE"


class OCRProcessingError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "OCR_PROCESSING_ERROR"


class ExtractionError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "EXTRACTION_ERROR"


class LLMTimeoutError(AppError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "LLM_TIMEOUT"


class DocumentNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "DOCUMENT_NOT_FOUND"


class DatabaseError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "DATABASE_ERROR"


def _error_body(request_id: str, code: str, message: str, details: dict) -> dict:
    return {
        "status": "error",
        "error_code": code,
        "message": message,
        "details": details,
        "request_id": request_id,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "Handled application error: %s | details=%s",
            exc.message,
            exc.details,
            extra={
                "request_id": request_id,
                "error_code": exc.error_code,
                "path": str(request.url),
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request_id, exc.error_code, exc.message, exc.details),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        # Full details go to the server log only - never to the client.
        logger.exception(
            "Unhandled exception", extra={"request_id": request_id, "path": str(request.url)}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                request_id,
                "INTERNAL_SERVER_ERROR",
                "An unexpected error occurred while processing the request.",
                {},
            ),
        )
