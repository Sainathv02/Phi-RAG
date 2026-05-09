"""Provider abstraction for chat backends."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Protocol, Tuple


class TurnLike(Protocol):
    question: str
    answer: str


class BaseProvider(ABC):
    """Uniform interface for local and external providers."""

    @abstractmethod
    def chat(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Tuple[str, int]:
        """Return full model response and latency in milliseconds."""

    @abstractmethod
    def stream(
        self,
        question: str,
        history: Optional[List[TurnLike]],
        context: str,
        model: str,
        api_key: Optional[str],
        profile,
    ) -> Iterator[str]:
        """Yield incremental model response chunks."""

    @abstractmethod
    def list_models(self, api_key: Optional[str] = None) -> List[str]:
        """List available models for this provider."""

    @abstractmethod
    def validate_key(self, api_key: str, model: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate provider API key when required."""
