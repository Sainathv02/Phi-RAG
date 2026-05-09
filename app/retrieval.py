"""Retrieval and context-building module."""

from typing import Any, Dict, List, Optional, Tuple

import legacy_app as legacy


RetrievedItem = Tuple[str, Dict[str, Any], float]


def tokenize_for_search(text: str) -> List[str]:
    """Tokenize text for lexical retrieval scoring."""

    return legacy.tokenize_for_search(text)


def adaptive_top_k(query: str) -> int:
    """Return adaptive TOP-K based on query complexity."""

    return legacy.get_adaptive_top_k(query)


def hybrid_retrieval(
    col,
    query: str,
    query_embedding: List[float],
    top_k: int,
    query_where: Optional[Dict[str, Any]] = None,
) -> List[RetrievedItem]:
    """Run hybrid semantic and BM25 retrieval with reranking."""

    return legacy.run_hybrid_retrieval(col, query, query_embedding, top_k, query_where=query_where)


def build_context(
    retrieved_items: List[RetrievedItem],
    collection_obj,
    max_chars: int,
) -> Tuple[str, List[dict]]:
    """Build bounded prompt context and source metadata list."""

    return legacy.build_context(retrieved_items, collection_obj, max_chars=max_chars)


def parse_source_filters(raw_filters: Optional[List[str]]) -> List[str]:
    """Normalize source filter list."""

    return legacy.parse_source_filters(raw_filters)


def build_source_where_clause(source_filters: List[str]) -> Optional[Dict[str, Any]]:
    """Build Chroma where clause from source filters."""

    return legacy.build_source_where_clause(source_filters)
