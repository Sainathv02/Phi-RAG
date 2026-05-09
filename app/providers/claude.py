"""Anthropic Claude provider implementation."""

from typing import Iterator, List, Optional, Tuple

import legacy_app as legacy

from .base import BaseProvider, TurnLike


class ClaudeProvider(BaseProvider):
    """Provider implementation for Claude models."""

    def chat(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Tuple[str, int]:
        return legacy.claude_chat(question, context, model, (api_key or "").strip(), history, profile)

    def stream(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Iterator[str]:
        return legacy.claude_chat_stream(question, context, model, (api_key or "").strip(), history, profile)

    def list_models(self, api_key: Optional[str] = None) -> List[str]:
        if api_key:
            models, _ = legacy.fetch_latest_claude_models(api_key)
            return models
        return list(legacy.CLAUDE_MODELS)

    def validate_key(self, api_key: str, model: Optional[str] = None):
        active_model = model or (legacy.CLAUDE_MODELS[0] if legacy.CLAUDE_MODELS else "")
        return legacy.validate_claude_key(api_key, active_model)
