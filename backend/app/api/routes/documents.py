from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.core.exceptions import FileValidationError
from app.db.database import get_db
from app.schemas.document import DocumentType, ProcessedDocumentSummary, ProcessingResult
from app.services import pipeline, storage_service

logger = logging.getLogger("document_intelligence")
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/process", response_model=ProcessingResult, status_code=status.HTTP_200_OK)
async def process_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, JPG or PNG file, max 3 pages"),
    document_type: DocumentType = Form(..., description="Document type selected in the frontend"),
    document_name: str | None = Form(
        None, description="Logical name to store/retrieve this document under. Defaults to the filename."
    ),
    db: Session = Depends(get_db),
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not file.filename:
        raise FileValidationError("No file was provided.")

    content = await file.read()
    resolved_name = document_name or file.filename

    logger.info(
        "Processing request received",
        extra={"request_id": request_id},
    )

    result = pipeline.process_document(
        document_name=resolved_name,
        document_type=document_type,
        filename=file.filename,
        content=content,
        request_id=request_id,
    )

    storage_service.save_processed_document(
        db=db,
        document_name=resolved_name,
        document_type=document_type.value,
        original_filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        result=result,
    )

    return result


@router.get("", response_model=list[ProcessedDocumentSummary])
def list_documents(latest_only: bool = True, limit: int = 200, db: Session = Depends(get_db)):
    """Backs the dashboard: document name, type, status, processed time."""
    return storage_service.list_processed_documents(db, latest_only=latest_only, limit=limit)


@router.get("/{document_name}", response_model=ProcessingResult)
def get_document_by_name(document_name: str, db: Session = Depends(get_db)):
    """Returns the latest processed result for the given document name."""
    row = storage_service.get_latest_by_name(db, document_name)
    return row.result_json
