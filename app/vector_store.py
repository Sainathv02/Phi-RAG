"""ChromaDB vector store operations module."""

from typing import Any, Dict, List, Optional, Tuple

import legacy_app as legacy

client = legacy.client
collection = legacy.collection


def ensure_compatible() -> Optional[str]:
    """Ensure collection embedding compatibility."""

    return legacy.ensure_collection_compatible()


def recreate_collection(name: str) -> None:
    """Recreate a collection preserving expected metadata."""

    legacy.recreate_collection(name)


def normalize_chat_id(chat_id: Optional[str]) -> str:
    """Normalize optional chat_id values for collection routing."""

    return legacy.normalize_chat_id(chat_id)


def chat_collection_name(chat_id: str) -> str:
    """Return collection name for a chat-scoped collection."""

    return legacy.chat_collection_name(chat_id)


def get_chat_collection(chat_id: str):
    """Return chat-scoped collection object."""

    return legacy.get_chat_collection(chat_id)


def list_sources() -> List[str]:
    """List distinct source filenames from the shared collection."""

    return legacy.list_indexed_sources()


def list_chat_collection_names() -> List[str]:
    """List all chat-scoped collection names."""

    return legacy.list_chat_collection_names()


def cleanup_orphans() -> None:
    """Delete orphaned chat collections with no matching session."""

    legacy.cleanup_orphaned_collections()


def hydrate_from_legacy(chat_id: str, source_filters: List[str]) -> int:
    """Hydrate a chat collection from legacy vectors for selected sources."""

    return legacy.hydrate_chat_collection_from_legacy(chat_id, source_filters)


def lookup_existing_file_index(file_hash: str, collection_obj=None) -> Tuple[bool, int]:
    """Check whether file hash already exists in target collection."""

    return legacy.lookup_existing_file_index(file_hash, collection_obj=collection_obj)


def delete_docs_for_file_hash(file_hash: str, collection_obj=None) -> int:
    """Delete indexed chunks by file hash."""

    return legacy.delete_docs_for_file_hash(file_hash, collection_obj=collection_obj)


def add_documents_in_batches(
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    collection_obj=None,
) -> None:
    """Add vectors in fixed-size batches."""

    legacy.add_documents_in_batches(ids, documents, embeddings, metadatas, collection_obj=collection_obj)
