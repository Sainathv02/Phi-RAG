"""SQLite history operations module."""

from typing import List, Optional

import legacy_app as legacy

ChatSession = legacy.ChatSession
ChatTurn = legacy.ChatTurn


def get_connection():
    """Return a configured SQLite connection."""

    return legacy.get_history_connection()


def init_schema() -> None:
    """Initialize and migrate the history schema."""

    legacy.init_history_db()


def migrate_legacy_history() -> None:
    """Migrate legacy JSONL history into SQLite when needed."""

    legacy.migrate_jsonl_history_if_needed()


def list_sessions(limit_chats: int = 20, limit_turns: int = 50) -> List[ChatSession]:
    """List persisted chat sessions with turns."""

    return legacy.list_chat_sessions(limit_chats=limit_chats, limit_turns=limit_turns)


def append_turn(chat_id: Optional[str], turn: ChatTurn, provider: str = "", model: str = "") -> str:
    """Persist a user/assistant turn to a session."""

    return legacy.append_turn_to_session(chat_id, turn, provider=provider, model=model)


def get_recent_turns(chat_id: Optional[str], limit_turns: int) -> List[ChatTurn]:
    """Return recent turns for contextual generation."""

    return legacy.get_recent_session_turns(chat_id, limit_turns)


def clear_history() -> int:
    """Clear all sessions and return deleted assistant turn count."""

    return legacy.clear_chat_history()


def delete_session(chat_id: str) -> dict:
    """Delete one chat session through the existing backend logic."""

    return legacy.delete_chat(chat_id)


def rename_session(chat_id: str, title: str) -> dict:
    """Rename a chat session title."""

    req = legacy.ChatRenameRequest(title=title)
    return legacy.rename_chat(chat_id, req)
