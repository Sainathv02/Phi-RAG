"""Provider registry for chat backends."""

from .base import BaseProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .local import LocalProvider
from .openai import OpenAIProvider

_local = LocalProvider()

PROVIDERS: dict[str, BaseProvider] = {
    "ollama": _local,
    "local": _local,
    "chatgpt": OpenAIProvider(),
    "gemini": GeminiProvider(),
    "claude": ClaudeProvider(),
}


def get_provider(name: str) -> BaseProvider:
    """Return provider instance by normalized name."""

    key = (name or "").strip().lower()
    if key not in PROVIDERS:
        raise KeyError(f"Unsupported provider: {name}")
    return PROVIDERS[key]
