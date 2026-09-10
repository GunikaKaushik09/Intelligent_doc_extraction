from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core import config as config_module
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@router.get("/health/llm-config")
def llm_config_status():
    """
    Diagnostics only -- never returns the key itself, just whether one is
    configured and which path a new /process request will actually take.
    Useful for confirming "is my API key actually being picked up" without
    digging through server logs or code.
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        key_configured = bool(settings.GEMINI_API_KEY)
        model = settings.GEMINI_MODEL
    elif provider == "anthropic":
        key_configured = bool(settings.ANTHROPIC_API_KEY)
        model = settings.LLM_MODEL
    else:
        key_configured = False
        model = None

    will_use_mock = settings.USE_MOCK_LLM or not key_configured

    return {
        "llm_provider": provider,
        "model": model,
        "api_key_configured": key_configured,
        "use_mock_llm_flag": settings.USE_MOCK_LLM,
        "next_request_will_use": "offline-document-aware-extractor" if will_use_mock else f"{provider} ({model})",
        "env_file_expected_at": str(config_module.ENV_FILE_PATH),
        "env_file_found": config_module.ENV_FILE_FOUND,
    }
