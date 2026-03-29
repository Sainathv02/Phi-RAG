import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests
from chromadb.errors import InvalidArgumentError
from docx import Document
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_FILE = DATA_DIR / "chat_history.jsonl"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE.touch(exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "phi4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "3200"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "400"))
PARENT_WINDOW_CHILD_SPAN = int(os.getenv("PARENT_WINDOW_CHILD_SPAN", "2"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBED_LEGACY_WORKERS = int(os.getenv("EMBED_LEGACY_WORKERS", "8"))
# Maximum token context passed to Ollama for embedding calls. Smaller values are faster
# for BERT-based models (attention is O(n²)). 512 covers chunks up to ~2000 chars.
EMBED_NUM_CTX = int(os.getenv("EMBED_NUM_CTX", "512"))
MAX_EMBED_NUM_CTX = int(os.getenv("MAX_EMBED_NUM_CTX", "2048"))
EMBED_CHARS_PER_TOKEN = int(os.getenv("EMBED_CHARS_PER_TOKEN", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.1"))
MIN_RETRIEVAL_RESULTS = int(os.getenv("MIN_RETRIEVAL_RESULTS", "2"))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", "0.9"))
CHAT_NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "128"))
CHAT_KEEP_ALIVE = os.getenv("CHAT_KEEP_ALIVE", "30m")
CHAT_REQUEST_TIMEOUT_SEC = int(os.getenv("CHAT_REQUEST_TIMEOUT_SEC", "420"))
MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "500"))
DOC_SNIPPET_CHARS = int(os.getenv("DOC_SNIPPET_CHARS", "900"))
INDEX_MAX_CHUNKS = int(os.getenv("INDEX_MAX_CHUNKS", "0"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
CHROMA_ADD_BATCH_SIZE = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "256"))

UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

COLLECTION_NAME = "rag_docs"
INDEX_SCHEMA_VERSION = "v2_parent_child"


def validate_runtime_config() -> None:
    if MAX_CHUNK_SIZE <= 0:
        raise ValueError("MAX_CHUNK_SIZE must be greater than 0.")
    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")
    if CHUNK_OVERLAP >= MAX_CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than MAX_CHUNK_SIZE.")
    if PARENT_CHUNK_SIZE <= 0:
        raise ValueError("PARENT_CHUNK_SIZE must be greater than 0.")
    if PARENT_CHUNK_OVERLAP < 0:
        raise ValueError("PARENT_CHUNK_OVERLAP cannot be negative.")
    if PARENT_CHUNK_OVERLAP >= PARENT_CHUNK_SIZE:
        raise ValueError("PARENT_CHUNK_OVERLAP must be smaller than PARENT_CHUNK_SIZE.")
    if PARENT_WINDOW_CHILD_SPAN < 0:
        raise ValueError("PARENT_WINDOW_CHILD_SPAN cannot be negative.")
    if EMBED_BATCH_SIZE <= 0:
        raise ValueError("EMBED_BATCH_SIZE must be greater than 0.")
    if EMBED_LEGACY_WORKERS <= 0:
        raise ValueError("EMBED_LEGACY_WORKERS must be greater than 0.")
    if EMBED_NUM_CTX <= 0:
        raise ValueError("EMBED_NUM_CTX must be greater than 0.")
    if MAX_EMBED_NUM_CTX < EMBED_NUM_CTX:
        raise ValueError("MAX_EMBED_NUM_CTX must be greater than or equal to EMBED_NUM_CTX.")
    if EMBED_CHARS_PER_TOKEN <= 0:
        raise ValueError("EMBED_CHARS_PER_TOKEN must be greater than 0.")
    if DEFAULT_TOP_K <= 0:
        raise ValueError("DEFAULT_TOP_K must be greater than 0.")
    if MIN_RETRIEVAL_RESULTS <= 0:
        raise ValueError("MIN_RETRIEVAL_RESULTS must be greater than 0.")
    if CHAT_REQUEST_TIMEOUT_SEC <= 0:
        raise ValueError("CHAT_REQUEST_TIMEOUT_SEC must be greater than 0.")
    if MAX_UPLOAD_MB <= 0:
        raise ValueError("MAX_UPLOAD_MB must be greater than 0.")
    if CHROMA_ADD_BATCH_SIZE <= 0:
        raise ValueError("CHROMA_ADD_BATCH_SIZE must be greater than 0.")


validate_runtime_config()

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
        "index_schema": INDEX_SCHEMA_VERSION,
    },
)

app = FastAPI(title="Phi Mini RAG Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

history_lock = threading.Lock()
upload_jobs_lock = threading.Lock()
upload_jobs: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    model: str
    model_response_ms: int
    total_response_ms: int
    chat_id: str


class ChatTurn(BaseModel):
    timestamp: str
    question: str
    answer: str
    sources: List[str]
    model: str
    model_response_ms: int
    total_response_ms: int


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    turns: List[ChatTurn]


def set_upload_job(job_id: str, **fields: Any) -> None:
    with upload_jobs_lock:
        current = upload_jobs.get(job_id, {})
        current.update(fields)
        current["updated_at"] = now_iso()
        upload_jobs[job_id] = current


def get_upload_job(job_id: str) -> Optional[Dict[str, Any]]:
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        return dict(job) if isinstance(job, dict) else None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history_store_unlocked() -> Dict[str, Any]:
    raw = HISTORY_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return {"sessions": []}

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
            return payload
    except Exception:
        pass

    # Legacy JSONL migration: fold old flat entries into a single session.
    turns: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        if "question" in entry and "answer" in entry:
            turns.append(
                {
                    "timestamp": str(entry.get("timestamp", now_iso())),
                    "question": str(entry.get("question", "")),
                    "answer": str(entry.get("answer", "")),
                    "sources": list(entry.get("sources", [])),
                    "model": str(entry.get("model", MODEL_NAME)),
                    "model_response_ms": int(entry.get("model_response_ms", 0)),
                    "total_response_ms": int(entry.get("total_response_ms", 0)),
                }
            )

    if not turns:
        return {"sessions": []}

    first_q = turns[0].get("question", "")
    legacy_session = {
        "id": str(uuid.uuid4()),
        "title": first_q[:60] if first_q else "Legacy Chat",
        "created_at": turns[0].get("timestamp", now_iso()),
        "updated_at": turns[-1].get("timestamp", now_iso()),
        "turns": turns,
    }
    return {"sessions": [legacy_session]}


def save_history_store_unlocked(store: Dict[str, Any]) -> None:
    HISTORY_FILE.write_text(json.dumps(store), encoding="utf-8")


def list_chat_sessions(limit_chats: int = 20, limit_turns: int = 50) -> List[ChatSession]:
    safe_chat_limit = max(1, min(limit_chats, MAX_HISTORY_ITEMS))
    safe_turn_limit = max(1, min(limit_turns, MAX_HISTORY_ITEMS))

    with history_lock:
        store = load_history_store_unlocked()

    sessions_payload = store.get("sessions", [])
    sessions: List[ChatSession] = []
    for payload in sessions_payload[:safe_chat_limit]:
        turns_payload = payload.get("turns", [])[-safe_turn_limit:]
        turns: List[ChatTurn] = []
        for t in turns_payload:
            try:
                turns.append(ChatTurn(**t))
            except Exception:
                continue

        try:
            sessions.append(
                ChatSession(
                    id=str(payload.get("id", "")),
                    title=str(payload.get("title", "Untitled Chat")),
                    created_at=str(payload.get("created_at", now_iso())),
                    updated_at=str(payload.get("updated_at", now_iso())),
                    turns=turns,
                )
            )
        except Exception:
            continue

    return sessions


def append_turn_to_session(chat_id: Optional[str], turn: ChatTurn) -> str:
    with history_lock:
        store = load_history_store_unlocked()
        sessions = store.get("sessions", [])

        target: Optional[Dict[str, Any]] = None
        if chat_id:
            for s in sessions:
                if str(s.get("id")) == chat_id:
                    target = s
                    break

        if target is None:
            target = {
                "id": str(uuid.uuid4()),
                "title": (turn.question[:60] if turn.question else "New Chat"),
                "created_at": turn.timestamp,
                "updated_at": turn.timestamp,
                "turns": [],
            }
            sessions.insert(0, target)

        target_turns = target.get("turns", [])
        target_turns.append(turn.model_dump())
        target["turns"] = target_turns
        target["updated_at"] = turn.timestamp
        if len(target_turns) == 1 and turn.question:
            target["title"] = turn.question[:60]

        target_id = str(target.get("id"))
        reordered = [s for s in sessions if str(s.get("id")) != target_id]
        reordered.insert(0, target)
        store["sessions"] = reordered[:MAX_HISTORY_ITEMS]
        save_history_store_unlocked(store)

    return target_id


def clear_chat_history() -> int:
    with history_lock:
        store = load_history_store_unlocked()
        sessions = store.get("sessions", [])
        deleted = sum(len(s.get("turns", [])) for s in sessions if isinstance(s, dict))
        HISTORY_FILE.write_text("", encoding="utf-8")
    return deleted


def is_dimension_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "dimension" in message and "embedding" in message


def recreate_collection() -> None:
    global collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # It's safe to continue if the collection does not exist yet.
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "index_schema": INDEX_SCHEMA_VERSION,
        },
    )


def get_collection_metadata() -> Dict[str, Any]:
    metadata = getattr(collection, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def ensure_collection_compatible() -> Optional[str]:
    metadata = get_collection_metadata()
    stored_model = metadata.get("embedding_model")
    if stored_model and stored_model != EMBEDDING_MODEL:
        recreate_collection()
        return (
            f"The persisted index was created with embedding model '{stored_model}' and was reset "
            f"for '{EMBEDDING_MODEL}'. Re-upload documents to rebuild the index."
        )
    return None


def save_upload_file(upload: UploadFile, destination: Path) -> Tuple[int, str]:
    total_bytes = 0
    file_hasher = hashlib.sha256()

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = upload.file.read(UPLOAD_STREAM_CHUNK_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Limit is {MAX_UPLOAD_MB} MB.",
                    )

                file_hasher.update(chunk)
                output_file.write(chunk)
    finally:
        upload.file.close()

    return total_bytes, file_hasher.hexdigest()


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def get_ollama_status(timeout_seconds: int = 5) -> Tuple[bool, List[str], Optional[str]]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, [], str(exc)

    payload = response.json()
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = [model.get("name", "") for model in models if isinstance(model, dict) and model.get("name")]
    return True, names, None


def lookup_existing_file_index(file_hash: str) -> Tuple[bool, int]:
    existing = collection.get(
        where={"$and": [{"file_hash": file_hash}, {"index_schema": INDEX_SCHEMA_VERSION}]},
        include=[],
    )
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    return (len(ids) > 0, len(ids))


def delete_docs_for_file_hash(file_hash: str) -> int:
    existing = collection.get(where={"file_hash": file_hash}, include=[])
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def index_uploaded_document(
    job_id: str,
    stored_path: Path,
    safe_name: str,
    file_hash: str,
    file_size_bytes: int,
    index_reset_message: Optional[str],
) -> None:
    upload_start = time.perf_counter()
    try:
        set_upload_job(job_id, status="running", message="Extracting text...")
        text = extract_text(stored_path)

        set_upload_job(job_id, status="running", message="Creating parent and child chunks...")
        hierarchical_chunks = build_parent_child_chunks(text)
        chunks = [record["text"] for record in hierarchical_chunks]

        if INDEX_MAX_CHUNKS > 0:
            hierarchical_chunks = hierarchical_chunks[:INDEX_MAX_CHUNKS]
            chunks = chunks[:INDEX_MAX_CHUNKS]

        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from this file.")

        delete_docs_for_file_hash(file_hash)

        set_upload_job(job_id, status="running", message=f"Embedding {len(chunks)} chunks...")
        embed_start = time.perf_counter()
        embeddings, unique_chunk_count, reused_chunk_count = embed_chunks_with_reuse(chunks)
        embedding_ms = int((time.perf_counter() - embed_start) * 1000)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for i, record in enumerate(hierarchical_chunks):
            metadatas.append(
                {
                    "source": safe_name,
                    "chunk": i,
                    "file_hash": file_hash,
                    "parent_id": record["parent_id"],
                    "parent_chunk": record["parent_chunk"],
                    "child_chunk": record["child_chunk"],
                    "index_schema": INDEX_SCHEMA_VERSION,
                }
            )

        set_upload_job(job_id, status="running", message="Writing vectors to index...")
        reset_performed = bool(index_reset_message)
        try:
            add_documents_in_batches(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        except InvalidArgumentError as exc:
            if not is_dimension_mismatch_error(exc):
                raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

            recreate_collection()
            add_documents_in_batches(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            reset_performed = True
            index_reset_message = (
                "The persisted index used a different embedding dimension and was rebuilt. "
                "Previously indexed documents must be uploaded again."
            )

        indexing_ms = int((time.perf_counter() - upload_start) * 1000)
        set_upload_job(
            job_id,
            status="completed",
            message="Document indexed successfully.",
            result={
                "message": "Document indexed successfully.",
                "file": safe_name,
                "file_size_bytes": file_size_bytes,
                "chunks": len(chunks),
                "parent_chunks": len({record["parent_id"] for record in hierarchical_chunks}),
                "unique_chunks": unique_chunk_count,
                "reused_chunk_embeddings": reused_chunk_count,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_ms": embedding_ms,
                "indexing_ms": indexing_ms,
                "index_reset": reset_performed,
                "index_reset_message": index_reset_message,
                "reused_existing_index": False,
                "file_hash": file_hash,
            },
        )
    except HTTPException as exc:
        set_upload_job(job_id, status="failed", message=exc.detail)
    except Exception as exc:
        set_upload_job(job_id, status="failed", message=f"Failed to process upload: {exc}")
    finally:
        safe_unlink(stored_path)


def embed_chunks_with_reuse(chunks: List[str]) -> Tuple[List[List[float]], int, int]:
    unique_chunks: List[str] = []
    unique_index_by_chunk: Dict[str, int] = {}
    chunk_indexes: List[int] = []
    reused_chunks = 0

    for chunk in chunks:
        existing_index = unique_index_by_chunk.get(chunk)
        if existing_index is None:
            existing_index = len(unique_chunks)
            unique_index_by_chunk[chunk] = existing_index
            unique_chunks.append(chunk)
        else:
            reused_chunks += 1

        chunk_indexes.append(existing_index)

    unique_embeddings = ollama_embed(unique_chunks)
    embeddings = [unique_embeddings[index] for index in chunk_indexes]
    return embeddings, len(unique_chunks), reused_chunks


def add_documents_in_batches(
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    for start in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
        end = start + CHROMA_ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)

    return chunks


def build_parent_child_chunks(text: str) -> List[Dict[str, Any]]:
    """Create larger parent chunks for coherence and smaller child chunks for retrieval."""
    parent_chunks = chunk_text(text, max_size=PARENT_CHUNK_SIZE, overlap=PARENT_CHUNK_OVERLAP)
    if not parent_chunks:
        return []

    records: List[Dict[str, Any]] = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        child_chunks = chunk_text(parent_text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not child_chunks:
            continue
        parent_id = f"p{parent_idx}"
        for child_idx, child_text in enumerate(child_chunks):
            records.append(
                {
                    "parent_id": parent_id,
                    "parent_chunk": parent_idx,
                    "child_chunk": child_idx,
                    "text": child_text,
                }
            )
    return records


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".docx":
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, MD, or DOCX.")


def _is_embed_context_error(message: str) -> bool:
    lowered = message.lower()
    return "input length exceeds the context length" in lowered or "context length" in lowered


def _truncate_for_embed_ctx(text: str, num_ctx: int) -> str:
    # Approximate tokenizer budget to avoid context-length hard failures.
    max_chars = max(32, num_ctx * EMBED_CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def ollama_embed_legacy_single(text: str, initial_num_ctx: int = EMBED_NUM_CTX) -> List[float]:
    num_ctx = initial_num_ctx

    while True:
        prompt_text = _truncate_for_embed_ctx(text, num_ctx)
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": prompt_text, "options": {"num_ctx": num_ctx}},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["embedding"]

        if _is_embed_context_error(resp.text) and num_ctx < MAX_EMBED_NUM_CTX:
            num_ctx = min(MAX_EMBED_NUM_CTX, num_ctx * 2)
            continue

        raise HTTPException(status_code=500, detail=f"Embedding failed: {resp.text}")


def ollama_embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    embeddings: List[List[float]] = []

    # Prefer batched embedding calls for better throughput; keep legacy fallback for compatibility.
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        num_ctx = EMBED_NUM_CTX
        resp = None

        while True:
            batch_for_ctx = [_truncate_for_embed_ctx(text, num_ctx) for text in batch]
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": batch_for_ctx, "options": {"num_ctx": num_ctx}},
                timeout=120,
            )

            if resp.status_code == 200:
                break

            if _is_embed_context_error(resp.text) and num_ctx < MAX_EMBED_NUM_CTX:
                num_ctx = min(MAX_EMBED_NUM_CTX, num_ctx * 2)
                continue

            break

        if resp.status_code == 200:
            data = resp.json()
            batch_embeddings = data.get("embeddings")
            if not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(batch):
                raise HTTPException(status_code=500, detail="Embedding failed: invalid embedding response shape.")
            embeddings.extend(batch_embeddings)
            continue

        # Concurrent fallback when /api/embed is unavailable on older Ollama builds.
        worker_count = max(1, min(EMBED_LEGACY_WORKERS, len(batch)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            batch_embeddings = list(executor.map(lambda t: ollama_embed_legacy_single(t, num_ctx), batch))
        embeddings.extend(batch_embeddings)

    return embeddings


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_context(
    retrieved_items: List[Tuple[str, Optional[dict], float]], max_chars: int = MAX_CONTEXT_CHARS
) -> Tuple[str, List[dict]]:
    selected = [i for i in retrieved_items if i[2] <= RETRIEVAL_MAX_DISTANCE]

    if len(selected) < MIN_RETRIEVAL_RESULTS:
        selected = retrieved_items[: max(1, MIN_RETRIEVAL_RESULTS)]

    context_parts: List[str] = []
    selected_metadatas: List[dict] = []
    total_chars = 0
    seen_parent_windows: set[str] = set()
    parent_children_cache: Dict[str, List[Tuple[int, str]]] = {}

    for _doc, metadata, distance in selected:
        md = metadata if isinstance(metadata, dict) else {}
        source = str(md.get("source", "unknown"))
        file_hash = str(md.get("file_hash", ""))
        parent_chunk = _safe_int(md.get("parent_chunk"), default=-1)
        child_chunk = _safe_int(md.get("child_chunk"), default=0)

        if file_hash and parent_chunk >= 0:
            parent_window_key = f"{file_hash}:{parent_chunk}:{child_chunk}"
            if parent_window_key in seen_parent_windows:
                continue
            seen_parent_windows.add(parent_window_key)

            parent_key = f"{file_hash}:{parent_chunk}"
            child_records = parent_children_cache.get(parent_key)
            if child_records is None:
                parent_rows = collection.get(
                    where={"$and": [{"file_hash": file_hash}, {"parent_chunk": parent_chunk}]},
                    include=["documents", "metadatas"],
                )
                parent_docs = parent_rows.get("documents", []) if isinstance(parent_rows, dict) else []
                parent_metas = parent_rows.get("metadatas", []) if isinstance(parent_rows, dict) else []

                child_records = []
                for idx, parent_doc in enumerate(parent_docs):
                    pmd = parent_metas[idx] if idx < len(parent_metas) and isinstance(parent_metas[idx], dict) else {}
                    child_idx = _safe_int(pmd.get("child_chunk"), default=idx)
                    child_records.append((child_idx, str(parent_doc).strip()))

                child_records.sort(key=lambda item: item[0])
                parent_children_cache[parent_key] = child_records

            if child_records:
                min_child = max(0, child_chunk - PARENT_WINDOW_CHILD_SPAN)
                max_child = child_chunk + PARENT_WINDOW_CHILD_SPAN
                window_docs = [doc for cidx, doc in child_records if min_child <= cidx <= max_child]
                if not window_docs:
                    window_docs = [doc for _, doc in child_records[: max(1, PARENT_WINDOW_CHILD_SPAN + 1)]]

                parent_window = "\n".join(window_docs)
                label = f"[source={source} parent={parent_chunk} child={child_chunk} distance={distance:.4f}]"
                block = f"{label}\n{parent_window[:DOC_SNIPPET_CHARS]}"
            else:
                block = (
                    f"[source={source} parent={parent_chunk} child={child_chunk} distance={distance:.4f}]\n"
                    f"{str(_doc).strip()[:DOC_SNIPPET_CHARS]}"
                )
        else:
            chunk = md.get("chunk", "?")
            block = f"[source={source} chunk={chunk} distance={distance:.4f}]\n{str(_doc).strip()[:DOC_SNIPPET_CHARS]}"

        if total_chars + len(block) > max_chars and context_parts:
            break

        context_parts.append(block)
        selected_metadatas.append(md)
        total_chars += len(block)

    trimmed_context = "\n\n".join(context_parts)
    if len(trimmed_context) > max_chars:
        trimmed_context = trimmed_context[:max_chars]

    return trimmed_context.strip(), selected_metadatas


def ollama_chat(question: str, context: str) -> Tuple[str, int]:
    system_prompt = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )

    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    model_start = time.perf_counter()
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "keep_alive": CHAT_KEEP_ALIVE,
        "options": {
            "temperature": CHAT_TEMPERATURE,
            "top_p": CHAT_TOP_P,
            "num_predict": CHAT_NUM_PREDICT,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
        )
    except requests.ReadTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Chat generation timed out after {CHAT_REQUEST_TIMEOUT_SEC}s. "
                "Try a shorter question/context or increase CHAT_REQUEST_TIMEOUT_SEC."
            ),
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama chat API: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {resp.text}")

    model_response_ms = int((time.perf_counter() - model_start) * 1000)
    data = resp.json()
    return data.get("message", {}).get("content", "No answer generated."), model_response_ms


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health(response: Response) -> dict:
    index_reset_message = ensure_collection_compatible()
    ollama_ok, available_models, ollama_error = get_ollama_status()
    collection_metadata = get_collection_metadata()
    collection_count = collection.count()

    status = "ok"
    issues: List[str] = []

    if index_reset_message:
        status = "degraded"
        issues.append(index_reset_message)

    if not ollama_ok:
        status = "degraded"
        issues.append(f"Ollama probe failed: {ollama_error}")

    if status != "ok":
        response.status_code = 503

    return {
        "status": status,
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "ollama": {
            "base_url": OLLAMA_BASE_URL,
            "reachable": ollama_ok,
            "available_models": available_models,
        },
        "index": {
            "collection": COLLECTION_NAME,
            "documents": collection_count,
            "metadata": collection_metadata,
        },
        "issues": issues,
    }


@app.get("/chat/history")
def chat_history(limit_chats: int = 20, limit_turns: int = 50) -> dict:
    sessions = list_chat_sessions(limit_chats=limit_chats, limit_turns=limit_turns)
    return {
        "sessions": [session.model_dump() for session in sessions],
        "count": len(sessions),
    }


@app.delete("/chat/history")
def delete_chat_history() -> dict:
    deleted = clear_chat_history()
    return {"message": "Chat history cleared.", "deleted_turns": deleted}


@app.post("/upload")
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    index_reset_message = ensure_collection_compatible()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    stored_path = UPLOAD_DIR / f"{file_id}_{safe_name}"
    file_size_bytes = 0

    try:
        file_size_bytes, file_hash = save_upload_file(file, stored_path)

        already_indexed, existing_chunks = lookup_existing_file_index(file_hash)
        if already_indexed:
            safe_unlink(stored_path)
            return {
                "message": "Document already indexed. Reused existing vectors.",
                "file": safe_name,
                "file_size_bytes": file_size_bytes,
                "chunks": existing_chunks,
                "unique_chunks": existing_chunks,
                "reused_chunk_embeddings": existing_chunks,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_ms": 0,
                "indexing_ms": 0,
                "index_reset": bool(index_reset_message),
                "index_reset_message": index_reset_message,
                "reused_existing_index": True,
                "file_hash": file_hash,
            }
    except HTTPException:
        safe_unlink(stored_path)
        raise
    except Exception as exc:
        safe_unlink(stored_path)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {exc}") from exc

    job_id = str(uuid.uuid4())
    set_upload_job(
        job_id,
        status="queued",
        file=safe_name,
        file_size_bytes=file_size_bytes,
        file_hash=file_hash,
        message="Upload saved. Indexing queued.",
    )
    background_tasks.add_task(
        index_uploaded_document,
        job_id,
        stored_path,
        safe_name,
        file_hash,
        file_size_bytes,
        index_reset_message,
    )

    return {
        "message": "Upload received. Indexing started in background.",
        "file": safe_name,
        "file_size_bytes": file_size_bytes,
        "job_id": job_id,
        "status": "queued",
        "reused_existing_index": False,
        "index_reset_message": index_reset_message,
    }


@app.get("/upload/jobs/{job_id}")
def upload_job_status(job_id: str) -> dict:
    job = get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return job


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    request_start = time.perf_counter()
    index_reset_message = ensure_collection_compatible()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    if index_reset_message:
        raise HTTPException(status_code=400, detail=index_reset_message)

    total_docs = collection.count()
    if total_docs == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload files first.")

    q_embedding = ollama_embed([req.question])[0]
    retrieval_k = max(1, req.top_k)

    try:
        result = collection.query(
            query_embeddings=[q_embedding],
            n_results=retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
    except InvalidArgumentError as exc:
        if is_dimension_mismatch_error(exc):
            recreate_collection()
            raise HTTPException(
                status_code=400,
                detail=(
                    "Embedding model dimension changed. Index was reset to match the current model. "
                    "Please upload documents again."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved_items = list(zip(docs, metadatas, distances))
    context, selected_metadatas = build_context(retrieved_items)
    answer, model_response_ms = ollama_chat(req.question, context)

    sources = []
    for m in selected_metadatas:
        src = m.get("source") if isinstance(m, dict) else None
        if src and src not in sources:
            sources.append(src)

    turn = ChatTurn(
        timestamp=now_iso(),
        question=req.question,
        answer=answer,
        sources=sources,
        model=MODEL_NAME,
        model_response_ms=model_response_ms,
        total_response_ms=int((time.perf_counter() - request_start) * 1000),
    )
    session_id = append_turn_to_session(req.chat_id, turn)
    return ChatResponse(
        answer=turn.answer,
        sources=turn.sources,
        model=turn.model,
        model_response_ms=turn.model_response_ms,
        total_response_ms=turn.total_response_ms,
        chat_id=session_id,
    )


@app.delete("/documents")
def delete_all_documents() -> dict:
    existing = collection.get(include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return {"message": "All indexed documents deleted.", "deleted": len(ids)}
