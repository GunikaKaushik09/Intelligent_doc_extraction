from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseError, DocumentNotFoundError
from app.db.models import ProcessedDocument
from app.schemas.document import ProcessedDocumentSummary, ProcessingResult


def save_processed_document(
    db: Session,
    document_name: str,
    document_type: str,
    original_filename: str,
    content_type: str,
    result: ProcessingResult,
) -> ProcessedDocument:
    try:
        row = ProcessedDocument(
            document_name=document_name,
            document_type=document_type,
            original_filename=original_filename,
            content_type=content_type,
            status=result.processing_status.value,
            result_json=json.loads(result.model_dump_json()),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    except Exception as exc:
        db.rollback()
        raise DatabaseError("Failed to persist processed document.", details={"reason": str(exc)}) from exc


def get_latest_by_name(db: Session, document_name: str) -> ProcessedDocument:
    stmt = (
        select(ProcessedDocument)
        .where(ProcessedDocument.document_name == document_name)
        .order_by(ProcessedDocument.created_at.desc())
        .limit(1)
    )
    row = db.execute(stmt).scalar_one_or_none()
    if row is None:
        raise DocumentNotFoundError(
            f"No processed document found with name '{document_name}'.",
            details={"document_name": document_name},
        )
    return row


def list_processed_documents(
    db: Session, latest_only: bool = True, limit: int = 200
) -> list[ProcessedDocumentSummary]:
    stmt = select(ProcessedDocument).order_by(ProcessedDocument.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()

    if latest_only:
        seen: set[str] = set()
        deduped = []
        for row in rows:
            if row.document_name in seen:
                continue
            seen.add(row.document_name)
            deduped.append(row)
        rows = deduped

    return [
        ProcessedDocumentSummary(
            id=row.id,
            document_name=row.document_name,
            document_type=row.document_type,
            status=row.status,
            created_at=row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc),
        )
        for row in rows
    ]
