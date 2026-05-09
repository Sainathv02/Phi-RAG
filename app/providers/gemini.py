"""Google Gemini provider implementation."""

from typing import Iterator, List, Optional, Tuple

import legacy_app as legacy

from .base import BaseProvider, TurnLike


class GeminiProvider(BaseProvider):
    """Provider implementation for Gemini models."""

    def chat(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Tuple[str, int]:
        return legacy.gemini_chat(question, context, model, (api_key or "").strip(), history, profile)

    def stream(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Iterator[str]:
        return legacy.gemini_chat_stream(question, context, model, (api_key or "").strip(), history, profile)

    def list_models(self, api_key: Optional[str] = None) -> List[str]:
        if api_key:
            models, _ = legacy.fetch_latest_gemini_models(api_key)
            return models
        return list(legacy.GEMINI_MODELS)

    def validate_key(self, api_key: str, model: Optional[str] = None):
        del model
        return legacy.validate_gemini_key(api_key)
