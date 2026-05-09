"""Routes for model listing and API-key validation."""

from fastapi import APIRouter

import legacy_app as legacy

router = APIRouter()


@router.get("/models")
def list_chat_models() -> dict:
    """Return local and external model lists."""

    return legacy.list_chat_models()


@router.post("/api-key/validate")
def validate_api_key_endpoint(req: legacy.ApiKeyValidateRequest) -> dict:
    """Validate provider API key and refresh provider models."""

    return legacy.validate_api_key_endpoint(req)
