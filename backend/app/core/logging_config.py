"""
Logging configuration.

Emits structured, single-line log records (timestamp, level, logger name,
request id when available, message) to stdout, which is what every free
hosting platform (Render, Railway, Fly.io, etc.) captures automatically.
"""
from __future__ import annotations

import logging
import sys


class RequestIdFilter(logging.Filter):
    """Ensures every record has a request_id field, even if not supplied."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(log_level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger("document_intelligence")
    root.setLevel(log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    # Quiet down noisy third-party loggers a little.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
