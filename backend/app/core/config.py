"""
Central application configuration.

All configurable values are read from environment variables so that no
secrets or environment-specific values are hardcoded. See `.env.example`
in the repo root for the full list of supported variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from a fixed location (backend/.env) rather than relying on
# python-dotenv's default cwd-search behaviour. That default silently loads
# whatever .env it finds first while walking up from the current working
# directory -- which depends entirely on where you happened to launch
# `uvicorn` from, and can find the wrong file (or none) without any error.
# Anchoring to this file's location makes it work the same way regardless
# of the working directory the server was started from.
ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
ENV_FILE_FOUND = ENV_FILE_PATH.is_file()
load_dotenv(dotenv_path=ENV_FILE_PATH)


class Settings:
    # --- General -----------------------------------------------------
    APP_NAME: str = "Document Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Storage -------------------------------------------------------
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/document_intelligence.db"
    )
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))

    # --- File validation limits -----------------------------------------
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
    MAX_PAGE_COUNT: int = int(os.getenv("MAX_PAGE_COUNT", "3"))
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".jpg", ".jpeg", ".png")
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "image/jpeg",
        "image/png",
    )

    # --- OCR -------------------------------------------------------------
    # Text-density threshold (chars/page) below which a PDF page is treated
    # as scanned/image-based and routed through OCR instead of native text
    # extraction.
    OCR_TEXT_DENSITY_THRESHOLD: int = int(
        os.getenv("OCR_TEXT_DENSITY_THRESHOLD", "40")
    )
    TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")  # optional override

    # --- LLM structuring ---------------------------------------------------
    # LLM_PROVIDER selects which backend does the OCR-text -> structured-JSON
    # step. "gemini" (default) uses Google's Gemini API, which has a genuine
    # no-credit-card-required free tier -- good for this assessment.
    # "anthropic" uses Claude if you have a paid API key. Either way, if no
    # key is configured for the selected provider, the app automatically
    # falls back to the offline heuristic extractor (see USE_MOCK_LLM).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    # When true (or when no API key is configured for the selected provider),
    # the extraction service falls back to a deterministic offline extractor
    # so the app remains demoable / testable without any paid API access.
    USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

    # --- Financial validation -----------------------------------------
    VALIDATION_TOLERANCE_PCT: float = float(
        os.getenv("VALIDATION_TOLERANCE_PCT", "0.01")
    )  # 1% relative tolerance, absorbs rounding differences

    # --- CORS -----------------------------------------------------------
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent.mkdir(
        parents=True, exist_ok=True
    ) if settings.DATABASE_URL.startswith("sqlite") else None
    return settings
