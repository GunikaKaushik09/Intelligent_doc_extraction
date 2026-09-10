from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProcessedDocument(Base):
    """
    One row per *processed result*. Re-processing a document with the same
    `document_name` inserts a new row; the GET-by-name endpoint returns the
    most recent row (highest `created_at`) for that name, per spec 5.1.
    Older rows are kept (versioning is optional but cheap to retain here).
    """

    __tablename__ = "processed_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    document_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, index=True, nullable=False)
    # Full structured JSON response (schemas.document.ProcessingResult) as dict.
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
