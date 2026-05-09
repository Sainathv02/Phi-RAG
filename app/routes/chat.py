"""Routes for chat request handling."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import legacy_app as legacy

router = APIRouter()


@router.post("/chat/stream")
def chat_stream(req: legacy.ChatRequest) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""

    return legacy.chat_stream(req)


@router.post("/chat", response_model=legacy.ChatResponse)
def chat(req: legacy.ChatRequest) -> legacy.ChatResponse:
    """Return non-streaming chat response."""

    return legacy.chat(req)
