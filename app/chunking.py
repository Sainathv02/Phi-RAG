"""Document extraction and chunking module."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import legacy_app as legacy


def extract_text(file_path: Path) -> str:
    """Extract text from supported file types."""

    return legacy.extract_text(file_path)


def semantic_chunk(text: str, max_chars: int = None, overlap_sentences: int = 2) -> List[str]:
    """Create semantic chunks preserving structure-heavy blocks."""

    if max_chars is None:
        max_chars = legacy.MAX_CHUNK_SIZE
    return legacy.semantic_chunk(text, max_chars=max_chars, overlap_sentences=overlap_sentences)


def build_parent_child_chunks(text: str) -> List[Dict[str, Any]]:
    """Build hierarchical parent-child chunk records."""

    return legacy.build_parent_child_chunks(text)


def embed_chunks_with_reuse(chunks: List[str]) -> Tuple[List[List[float]], int, int]:
    """Embed chunks while deduplicating repeated text."""

    return legacy.embed_chunks_with_reuse(chunks)
