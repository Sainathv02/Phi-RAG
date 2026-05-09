"""Local Ollama provider implementation."""

from typing import Iterator, List, Optional, Tuple

import legacy_app as legacy

from .base import BaseProvider, TurnLike


class LocalProvider(BaseProvider):
    """Provider implementation backed by Ollama local runtime."""

    def chat(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Tuple[str, int]:
        del api_key
        return legacy.ollama_chat(question, context, model, history, profile)

    def stream(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Iterator[str]:
        del api_key
        return legacy.ollama_chat_stream(question, context, model, history, profile)

    def list_models(self, api_key: Optional[str] = None) -> List[str]:
        del api_key
        return list(legacy.CHAT_MODEL_OPTIONS)

    def validate_key(self, api_key: str, model: Optional[str] = None):
        del api_key, model
        return True, None
