"""Routes for chat history and session lifecycle."""

from fastapi import APIRouter

import legacy_app as legacy

router = APIRouter()


@router.get("/chat/history")
def chat_history(limit_chats: int = 20, limit_turns: int = 50) -> dict:
    """List historical chat sessions and turns."""

    return legacy.chat_history(limit_chats=limit_chats, limit_turns=limit_turns)


@router.delete("/chat/history")
def delete_chat_history() -> dict:
    """Clear all chat sessions."""

    return legacy.delete_chat_history()


@router.delete("/chat/{chat_id}")
def delete_chat(chat_id: str) -> dict:
    """Delete one chat session."""

    return legacy.delete_chat(chat_id)


@router.patch("/chat/{chat_id}/title")
def rename_chat(chat_id: str, req: legacy.ChatRenameRequest) -> dict:
    """Rename a chat session title."""

    return legacy.rename_chat(chat_id, req)
