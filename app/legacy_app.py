import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import chromadb
import requests
from chromadb.errors import InvalidArgumentError
from docx import Document
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile
from fastapi import Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_FILE = DATA_DIR / "chat_history.jsonl"
HISTORY_DB_FILE = DATA_DIR / "chat_history.sqlite3"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE.touch(exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "phi4-mini:3.8b-q4_K_M")
CHAT_MODEL_OPTIONS_RAW = os.getenv(
    "CHAT_MODEL_OPTIONS",
    "phi4-mini:3.8b-q4_K_M,qwen2.5:3b-instruct-q4_K_M,qwen2.5:1.5b",
)
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
CHAT_NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "2048"))
CHAT_KEEP_ALIVE = os.getenv("CHAT_KEEP_ALIVE", "30m")
CHAT_REQUEST_TIMEOUT_SEC = int(os.getenv("CHAT_REQUEST_TIMEOUT_SEC", "420"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
LOCAL_CHAT_NUM_PREDICT = int(os.getenv("LOCAL_CHAT_NUM_PREDICT", str(CHAT_NUM_PREDICT)))
API_CHAT_NUM_PREDICT = int(os.getenv("API_CHAT_NUM_PREDICT", str(max(CHAT_NUM_PREDICT * 4, 512))))
LOCAL_MAX_CONTEXT_CHARS = int(os.getenv("LOCAL_MAX_CONTEXT_CHARS", str(MAX_CONTEXT_CHARS)))
API_MAX_CONTEXT_CHARS = int(os.getenv("API_MAX_CONTEXT_CHARS", str(max(MAX_CONTEXT_CHARS * 4, 12000))))
LOCAL_CHAT_HISTORY_TURNS = int(os.getenv("LOCAL_CHAT_HISTORY_TURNS", str(max(2, CHAT_HISTORY_TURNS - 2))))
API_CHAT_HISTORY_TURNS = int(os.getenv("API_CHAT_HISTORY_TURNS", str(max(CHAT_HISTORY_TURNS + 4, 10))))
MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "500"))
DOC_SNIPPET_CHARS = int(os.getenv("DOC_SNIPPET_CHARS", "900"))
INDEX_MAX_CHUNKS = int(os.getenv("INDEX_MAX_CHUNKS", "0"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
CHROMA_ADD_BATCH_SIZE = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "256"))

UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

COLLECTION_NAME = "rag_docs"
CHAT_COLLECTION_PREFIX = "chat_"
INDEX_SCHEMA_VERSION = "v2_parent_child"

# External model provider definitions (static — not pulled from Ollama)
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]
CLAUDE_MODELS = [
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-haiku-3-5",
]
CHATGPT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

API_PROVIDERS = {"chatgpt", "gemini", "claude"}


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    num_predict: int
    max_context_chars: int
    history_turns: int


LOCAL_GENERATION_PROFILE = GenerationProfile(
    name="local",
    num_predict=max(32, LOCAL_CHAT_NUM_PREDICT),
    max_context_chars=max(1200, LOCAL_MAX_CONTEXT_CHARS),
    history_turns=max(0, LOCAL_CHAT_HISTORY_TURNS),
)

API_GENERATION_PROFILE = GenerationProfile(
    name="api",
    num_predict=max(64, API_CHAT_NUM_PREDICT),
    max_context_chars=max(2400, API_MAX_CONTEXT_CHARS),
    history_turns=max(0, API_CHAT_HISTORY_TURNS),
)


def parse_model_options(raw_value: str) -> List[str]:
    models: List[str] = []
    for token in raw_value.split(","):
        model = token.strip()
        if model and model not in models:
            models.append(model)
    return models


CHAT_MODEL_OPTIONS = parse_model_options(CHAT_MODEL_OPTIONS_RAW)
if MODEL_NAME not in CHAT_MODEL_OPTIONS:
    CHAT_MODEL_OPTIONS.insert(0, MODEL_NAME)


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
    if CHAT_HISTORY_TURNS < 0:
        raise ValueError("CHAT_HISTORY_TURNS cannot be negative.")
    if LOCAL_CHAT_NUM_PREDICT <= 0:
        raise ValueError("LOCAL_CHAT_NUM_PREDICT must be greater than 0.")
    if API_CHAT_NUM_PREDICT <= 0:
        raise ValueError("API_CHAT_NUM_PREDICT must be greater than 0.")
    if LOCAL_MAX_CONTEXT_CHARS <= 0:
        raise ValueError("LOCAL_MAX_CONTEXT_CHARS must be greater than 0.")
    if API_MAX_CONTEXT_CHARS <= 0:
        raise ValueError("API_MAX_CONTEXT_CHARS must be greater than 0.")
    if LOCAL_CHAT_HISTORY_TURNS < 0:
        raise ValueError("LOCAL_CHAT_HISTORY_TURNS cannot be negative.")
    if API_CHAT_HISTORY_TURNS < 0:
        raise ValueError("API_CHAT_HISTORY_TURNS cannot be negative.")
    if MAX_UPLOAD_MB <= 0:
        raise ValueError("MAX_UPLOAD_MB must be greater than 0.")
    if CHROMA_ADD_BATCH_SIZE <= 0:
        raise ValueError("CHROMA_ADD_BATCH_SIZE must be greater than 0.")


def _scaled_int(value: int, factor: float, floor: int) -> int:
    return max(floor, int(round(value * factor)))


def _adjust_profile_for_model(provider: str, model_name: str, base: GenerationProfile) -> GenerationProfile:
    model = (model_name or "").strip().lower()
    if not model:
        return base

    num_predict = base.num_predict
    max_context_chars = base.max_context_chars

    # Smaller/fast API variants usually benefit from shorter outputs and context windows.
    if provider in API_PROVIDERS and any(tag in model for tag in ("mini", "flash", "haiku")):
        num_predict = _scaled_int(num_predict, 0.75, 192)
        max_context_chars = _scaled_int(max_context_chars, 0.7, 5000)
    elif provider in API_PROVIDERS and any(tag in model for tag in ("pro", "opus", "sonnet", "gpt-4o", "gpt-4.1")):
        num_predict = _scaled_int(num_predict, 1.1, base.num_predict)
        max_context_chars = _scaled_int(max_context_chars, 1.1, base.max_context_chars)

    return GenerationProfile(
        name=base.name,
        num_predict=num_predict,
        max_context_chars=max_context_chars,
        history_turns=base.history_turns,
    )


def resolve_generation_profile(provider: str, model_name: str) -> GenerationProfile:
    normalized = (provider or "").strip().lower()
    if normalized == "local":
        normalized = "ollama"

    base = API_GENERATION_PROFILE if normalized in API_PROVIDERS else LOCAL_GENERATION_PROFILE
    resolved = _adjust_profile_for_model(normalized, model_name, base)
    profile_label = "API" if resolved.name == "api" else "LOCAL"
    provider_label = "local" if normalized == "ollama" else normalized
    logger.info(
        "Using %s profile for provider: %s (model=%s, num_predict=%d, max_context_chars=%d, history_turns=%d)",
        profile_label,
        provider_label,
        model_name,
        resolved.num_predict,
        resolved.max_context_chars,
        resolved.history_turns,
    )
    return resolved


def external_max_tokens(profile: GenerationProfile) -> int:
    return max(128, profile.num_predict * 4)


validate_runtime_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")

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
bm25_cache_lock = threading.Lock()
bm25_cache: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    chat_id: Optional[str] = None
    model: Optional[str] = None
    source_filters: Optional[List[str]] = None
    # External provider support. provider must be "ollama" | "gemini" | "claude" | "chatgpt".
    # api_key is never stored on the server; it is used only for the duration of this request.
    provider: str = "ollama"
    api_key: Optional[str] = None


class ApiKeyValidateRequest(BaseModel):
    provider: str  # "gemini" | "claude"
    model: str
    api_key: str


class ChatRenameRequest(BaseModel):
    title: str


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
    provider: Optional[str] = None
    model: Optional[str] = None
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


def get_history_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(HISTORY_DB_FILE), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_history_db() -> None:
    with get_history_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              title TEXT,
              provider TEXT,
              model TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Migrate existing sessions tables that lack provider/model columns
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "provider" not in existing_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN provider TEXT")
        if "model" not in existing_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT REFERENCES sessions(id),
              role TEXT CHECK(role IN ('user', 'assistant')),
              content TEXT,
              sources TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at, id);")


def load_legacy_history_sessions() -> List[Dict[str, Any]]:
    raw = HISTORY_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
            return payload.get("sessions", [])
    except Exception:
        pass

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
        return []

    first_q = turns[0].get("question", "")
    return [
        {
            "id": str(uuid.uuid4()),
            "title": first_q[:60] if first_q else "Legacy Chat",
            "created_at": turns[0].get("timestamp", now_iso()),
            "updated_at": turns[-1].get("timestamp", now_iso()),
            "turns": turns,
        }
    ]


def migrate_jsonl_history_if_needed() -> None:
    if not HISTORY_FILE.exists():
        return
    raw = HISTORY_FILE.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return

    with get_history_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
        if row and int(row["c"]) > 0:
            return

    sessions = load_legacy_history_sessions()
    if not sessions:
        return

    with history_lock:
        with get_history_connection() as conn:
            try:
                conn.execute("BEGIN")
                for s in sessions:
                    session_id = str(s.get("id") or uuid.uuid4())
                    created_at = str(s.get("created_at") or now_iso())
                    updated_at = str(s.get("updated_at") or created_at)
                    title = str(s.get("title") or "Untitled Chat")

                    conn.execute(
                        "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (session_id, title, created_at, updated_at),
                    )

                    for t in s.get("turns", []):
                        ts = str(t.get("timestamp") or now_iso())
                        question = str(t.get("question") or "")
                        answer = str(t.get("answer") or "")
                        sources = t.get("sources") if isinstance(t.get("sources"), list) else []
                        conn.execute(
                            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, 'user', ?, ?, ?)",
                            (session_id, question, "[]", ts),
                        )
                        conn.execute(
                            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                            (session_id, answer, json.dumps(sources), ts),
                        )

                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise


def _messages_to_turns(rows: List[sqlite3.Row], limit_turns: int) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    pending_question = ""
    pending_ts = now_iso()

    for row in rows:
        role = str(row["role"] or "")
        content = str(row["content"] or "")
        created_at = str(row["created_at"] or now_iso())

        if role == "user":
            pending_question = content
            pending_ts = created_at
            continue

        if role != "assistant":
            continue

        try:
            parsed_sources = json.loads(row["sources"] or "[]")
            sources = parsed_sources if isinstance(parsed_sources, list) else []
        except Exception:
            sources = []

        turns.append(
            ChatTurn(
                timestamp=created_at or pending_ts,
                question=pending_question,
                answer=content,
                sources=[str(x) for x in sources],
                model="",
                model_response_ms=0,
                total_response_ms=0,
            )
        )
        pending_question = ""
        pending_ts = now_iso()

    if limit_turns > 0:
        return turns[-limit_turns:]
    return turns


def list_chat_sessions(limit_chats: int = 20, limit_turns: int = 50) -> List[ChatSession]:
    safe_chat_limit = max(1, min(limit_chats, MAX_HISTORY_ITEMS))
    safe_turn_limit = max(1, min(limit_turns, MAX_HISTORY_ITEMS))

    with history_lock:
        with get_history_connection() as conn:
            session_rows = conn.execute(
                "SELECT id, title, provider, model, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (safe_chat_limit,),
            ).fetchall()

            sessions: List[ChatSession] = []
            for row in session_rows:
                msg_rows = conn.execute(
                    "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                    (row["id"],),
                ).fetchall()
                turns = _messages_to_turns(msg_rows, safe_turn_limit)
                sessions.append(
                    ChatSession(
                        id=str(row["id"]),
                        title=str(row["title"] or "Untitled Chat"),
                        provider=row["provider"] or None,
                        model=row["model"] or None,
                        created_at=str(row["created_at"] or now_iso()),
                        updated_at=str(row["updated_at"] or now_iso()),
                        turns=turns,
                    )
                )

            return sessions


def append_turn_to_session(chat_id: Optional[str], turn: ChatTurn, provider: str = "", model: str = "") -> str:
    with history_lock:
        with get_history_connection() as conn:
            try:
                conn.execute("BEGIN")

                session_id = chat_id.strip() if isinstance(chat_id, str) and chat_id.strip() else ""
                row = None
                if session_id:
                    row = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()

                if not row:
                    session_id = session_id or str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO sessions (id, title, provider, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, (turn.question[:60] if turn.question else "New Chat"), provider or None, model or None, turn.timestamp, turn.timestamp),
                    )

                msg_count_row = conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND role = 'assistant'",
                    (session_id,),
                ).fetchone()
                is_first_turn = int(msg_count_row["c"]) == 0 if msg_count_row else False

                conn.execute(
                    "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, 'user', ?, ?, ?)",
                    (session_id, turn.question, "[]", turn.timestamp),
                )
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                    (session_id, turn.answer, json.dumps(turn.sources), turn.timestamp),
                )

                if is_first_turn and turn.question:
                    conn.execute(
                        "UPDATE sessions SET title = ?, provider = ?, model = ?, updated_at = ? WHERE id = ?",
                        (turn.question[:60], provider or None, model or None, turn.timestamp, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE sessions SET provider = ?, model = ?, updated_at = ? WHERE id = ?",
                        (provider or None, model or None, turn.timestamp, session_id),
                    )

                conn.execute("COMMIT")
                return session_id
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=500, detail=f"Failed to save chat history: {exc}") from exc


def get_recent_session_turns(chat_id: Optional[str], limit_turns: int) -> List[ChatTurn]:
    if not chat_id or limit_turns <= 0:
        return []

    with history_lock:
        with get_history_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                (chat_id,),
            ).fetchall()
    return _messages_to_turns(rows, limit_turns)


def clear_chat_history() -> int:
    session_ids: List[str] = []
    with history_lock:
        with get_history_connection() as conn:
            try:
                conn.execute("BEGIN")
                session_rows = conn.execute("SELECT id FROM sessions").fetchall()
                session_ids = [str(row["id"]).strip() for row in session_rows if row and row["id"]]
                deleted_row = conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE role = 'assistant'"
                ).fetchone()
                deleted = int(deleted_row["c"]) if deleted_row else 0
                conn.execute("DELETE FROM messages")
                conn.execute("DELETE FROM sessions")
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {exc}") from exc

    for sid in session_ids:
        try:
            cname = chat_collection_name(sid)
            client.delete_collection(name=cname)
            with bm25_cache_lock:
                bm25_cache.pop(cname, None)
        except Exception:
            pass
    return deleted


init_history_db()
migrate_jsonl_history_if_needed()


def is_dimension_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "dimension" in message and "embedding" in message


def recreate_collection(collection_name: str = COLLECTION_NAME) -> None:
    global collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # It's safe to continue if the collection does not exist yet.
        pass

    recreated = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "index_schema": INDEX_SCHEMA_VERSION,
        },
    )
    if collection_name == COLLECTION_NAME:
        collection = recreated
    with bm25_cache_lock:
        bm25_cache.pop(collection_name, None)


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


def normalize_chat_id(chat_id: Optional[str]) -> str:
    if not isinstance(chat_id, str):
        return ""
    return chat_id.strip()


def chat_collection_name(chat_id: str) -> str:
    return f"{CHAT_COLLECTION_PREFIX}{chat_id}"


def get_chat_collection(chat_id: str):
    safe_chat_id = normalize_chat_id(chat_id)
    if not safe_chat_id:
        return collection
    return client.get_or_create_collection(
        name=chat_collection_name(safe_chat_id),
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "index_schema": INDEX_SCHEMA_VERSION,
            "chat_id": safe_chat_id,
        },
    )


def list_chat_collection_names() -> List[str]:
    names: List[str] = []
    try:
        for col in client.list_collections():
            col_name = getattr(col, "name", "")
            if isinstance(col_name, str) and col_name.startswith(CHAT_COLLECTION_PREFIX):
                names.append(col_name)
    except Exception:
        return []
    return names


def get_all_session_ids() -> set[str]:
    with history_lock:
        with get_history_connection() as conn:
            rows = conn.execute("SELECT id FROM sessions").fetchall()
    return {str(row["id"]).strip() for row in rows if row and row["id"]}


def cleanup_orphaned_collections() -> None:
    valid_sessions = get_all_session_ids()
    for col_name in list_chat_collection_names():
        chat_id = col_name[len(CHAT_COLLECTION_PREFIX) :]
        if chat_id and chat_id not in valid_sessions:
            try:
                client.delete_collection(name=col_name)
                logger.info("[cleanup] deleted orphaned collection=%s", col_name)
            except Exception as exc:
                logger.warning("[cleanup] failed to delete orphaned collection=%s error=%s", col_name, exc)


def migrate_legacy_collection_to_existing_sessions() -> None:
    """Best-effort migration from shared legacy collection into session-scoped collections."""
    try:
        legacy_count = int(collection.count() or 0)
    except Exception:
        legacy_count = 0
    if legacy_count <= 0:
        return

    session_sources: Dict[str, set[str]] = {}
    with history_lock:
        with get_history_connection() as conn:
            rows = conn.execute(
                "SELECT session_id, sources FROM messages WHERE role = 'assistant'"
            ).fetchall()

    for row in rows:
        sid = str(row["session_id"] or "").strip()
        if not sid:
            continue
        try:
            parsed = json.loads(row["sources"] or "[]")
        except Exception:
            parsed = []
        if not isinstance(parsed, list):
            continue
        bucket = session_sources.setdefault(sid, set())
        for src in parsed:
            value = str(src or "").strip()
            if value:
                bucket.add(value)

    moved_total = 0
    for sid, sources in session_sources.items():
        if not sources:
            continue
        moved_total += hydrate_chat_collection_from_legacy(sid, sorted(sources))

    if moved_total:
        logger.info("[migration] moved %s legacy chunks into chat-scoped collections", moved_total)

def tokenize_for_search(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower())


def get_adaptive_top_k(query: str) -> int:
    text = (query or "").strip().lower()
    words = [w for w in re.split(r"\s+", text) if w]
    phrase_set = {
        "compare",
        "list all",
        "summarize everything",
        "difference between",
        "explain all",
    }
    medium_hints = {"explain", "describe", "how does"}

    is_complex = len(words) > 20 or any(p in text for p in phrase_set)
    is_medium = (8 <= len(words) <= 20) or any(p in text for p in medium_hints)

    if is_complex:
        logger.info("[retrieval] Query complexity: complex -> TOP_K=10")
        return 10
    if is_medium:
        logger.info("[retrieval] Query complexity: medium -> TOP_K=6")
        return 6
    logger.info("[retrieval] Query complexity: simple -> TOP_K=3")
    return 3


def get_bm25_index(col) -> Optional[Dict[str, Any]]:
    col_name = str(getattr(col, "name", ""))
    if not col_name:
        return None

    doc_count = int(col.count() or 0)
    if doc_count <= 0:
        return None

    with bm25_cache_lock:
        cached = bm25_cache.get(col_name)
        if cached and int(cached.get("doc_count", -1)) == doc_count:
            return cached

    rows = col.get(include=["documents", "metadatas"])
    ids = rows.get("ids", []) if isinstance(rows, dict) else []
    docs = rows.get("documents", []) if isinstance(rows, dict) else []
    metas = rows.get("metadatas", []) if isinstance(rows, dict) else []
    if not ids or not docs:
        return None

    tokenized = [tokenize_for_search(str(doc)) for doc in docs]
    if not any(tokenized):
        return None

    payload = {
        "doc_count": doc_count,
        "ids": ids,
        "docs": docs,
        "metas": metas,
        "bm25": BM25Okapi(tokenized),
    }
    with bm25_cache_lock:
        bm25_cache[col_name] = payload
    return payload


def reciprocal_rank_fusion(semantic_ids: List[str], bm25_ids: List[str], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(semantic_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_terms = set(tokenize_for_search(query))
    scored: List[Dict[str, Any]] = []
    for item in chunks:
        doc_text = str(item.get("doc") or "")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        doc_terms = set(tokenize_for_search(doc_text))
        overlap = 0.0
        if query_terms:
            overlap = len(query_terms.intersection(doc_terms)) / max(1, len(query_terms))

        distance = float(item.get("distance") if item.get("distance") is not None else 1.5)
        semantic_score = 1.0 / (1.0 + max(0.0, distance))

        chunk_pos = _safe_int(metadata.get("chunk"), default=999)
        if chunk_pos >= 999:
            chunk_pos = _safe_int(metadata.get("child_chunk"), default=999)
        position_score = max(0.0, 1.0 - (min(50, max(0, chunk_pos)) / 50.0))

        final_score = (0.5 * semantic_score) + (0.3 * overlap) + (0.2 * position_score)
        copy_item = dict(item)
        copy_item["final_score"] = final_score
        scored.append(copy_item)

    scored.sort(key=lambda row: row.get("final_score", 0.0), reverse=True)
    logger.info("[rerank] Top chunks: %s", [round(float(r.get("final_score", 0.0)), 4) for r in scored[:3]])
    return scored


def run_hybrid_retrieval(
    col,
    query: str,
    query_embedding: List[float],
    top_k: int,
    query_where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    semantic_fetch_k = max(top_k * 3, top_k)

    def semantic_search() -> Dict[str, Any]:
        return col.query(
            query_embeddings=[query_embedding],
            n_results=semantic_fetch_k,
            include=["documents", "metadatas", "distances"],
            where=query_where,
        )

    def bm25_search() -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        bm25_ranked: List[str] = []
        bm25_items: Dict[str, Dict[str, Any]] = {}
        bm25_payload = get_bm25_index(col)
        tokens = tokenize_for_search(query)
        if not bm25_payload or not tokens:
            return bm25_ranked, bm25_items

        source_filter_set: set[str] = set()
        if isinstance(query_where, dict):
            if "source" in query_where and isinstance(query_where.get("source"), str):
                source_filter_set.add(str(query_where.get("source")))
            source_in = query_where.get("source") if isinstance(query_where.get("source"), dict) else {}
            if isinstance(source_in, dict) and isinstance(source_in.get("$in"), list):
                source_filter_set.update(str(x) for x in source_in.get("$in") if str(x).strip())

        scores = bm25_payload["bm25"].get_scores(tokens)
        all_ids = bm25_payload.get("ids", [])
        all_docs = bm25_payload.get("docs", [])
        all_metas = bm25_payload.get("metas", [])
        candidate_rows: List[Tuple[str, float, str, Dict[str, Any]]] = []
        for idx, score in enumerate(scores):
            if idx >= len(all_ids):
                continue
            md = all_metas[idx] if idx < len(all_metas) and isinstance(all_metas[idx], dict) else {}
            if source_filter_set:
                src = str(md.get("source") or "")
                if src not in source_filter_set:
                    continue
            candidate_rows.append((str(all_ids[idx]), float(score), str(all_docs[idx]) if idx < len(all_docs) else "", md))

        candidate_rows.sort(key=lambda row: row[1], reverse=True)
        for doc_id, score, doc, md in candidate_rows[:semantic_fetch_k]:
            if score <= 0:
                continue
            bm25_ranked.append(doc_id)
            bm25_items[doc_id] = {
                "id": doc_id,
                "doc": doc,
                "metadata": md,
                "distance": None,
            }
        return bm25_ranked, bm25_items

    with ThreadPoolExecutor(max_workers=2) as executor:
        sem_future = executor.submit(semantic_search)
        bm_future = executor.submit(bm25_search)
        result = sem_future.result()
        bm25_ranked_ids, bm25_hits = bm_future.result()

    semantic_docs = result.get("documents", [[]])[0]
    semantic_metas = result.get("metadatas", [[]])[0]
    semantic_dists = result.get("distances", [[]])[0]
    semantic_ids = result.get("ids", [[]])[0]

    semantic_hits: Dict[str, Dict[str, Any]] = {}
    semantic_ranked_ids: List[str] = []
    for idx, doc_id in enumerate(semantic_ids):
        sid = str(doc_id)
        semantic_ranked_ids.append(sid)
        semantic_hits[sid] = {
            "id": sid,
            "doc": str(semantic_docs[idx]) if idx < len(semantic_docs) else "",
            "metadata": semantic_metas[idx] if idx < len(semantic_metas) and isinstance(semantic_metas[idx], dict) else {},
            "distance": float(semantic_dists[idx]) if idx < len(semantic_dists) else None,
        }

    fused_ids = reciprocal_rank_fusion(semantic_ranked_ids, bm25_ranked_ids)
    if not fused_ids:
        fused_ids = semantic_ranked_ids

    fused_chunks: List[Dict[str, Any]] = []
    for doc_id in fused_ids:
        if doc_id in semantic_hits:
            fused_chunks.append(semantic_hits[doc_id])
        elif doc_id in bm25_hits:
            fused_chunks.append(bm25_hits[doc_id])

    reranked = rerank_chunks(query, fused_chunks)
    selected = reranked[:top_k]
    return [
        (
            str(item.get("doc") or ""),
            item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            float(item.get("distance")) if item.get("distance") is not None else 1.5,
        )
        for item in selected
    ]


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


def list_indexed_sources() -> List[str]:
    rows = collection.get(include=["metadatas"])
    metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []
    names: List[str] = []
    seen: set[str] = set()

    for md in metadatas:
        if not isinstance(md, dict):
            continue
        source = str(md.get("source", "")).strip()
        if not source or source in seen:
            continue
        seen.add(source)
        names.append(source)

    names.sort(key=lambda s: s.lower())
    return names


def lookup_existing_file_index(file_hash: str, collection_obj=None) -> Tuple[bool, int]:
    col = collection_obj or collection
    existing = col.get(
        where={"$and": [{"file_hash": file_hash}, {"index_schema": INDEX_SCHEMA_VERSION}]},
        include=[],
    )
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    return (len(ids) > 0, len(ids))


def delete_docs_for_file_hash(file_hash: str, collection_obj=None) -> int:
    col = collection_obj or collection
    existing = col.get(where={"file_hash": file_hash}, include=[])
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if ids:
        col.delete(ids=ids)
        with bm25_cache_lock:
            bm25_cache.pop(str(getattr(col, "name", "")), None)
    return len(ids)


def hydrate_chat_collection_from_legacy(chat_id: str, source_filters: List[str]) -> int:
    safe_chat_id = normalize_chat_id(chat_id)
    if not safe_chat_id or not source_filters:
        return 0

    target_collection = get_chat_collection(safe_chat_id)
    existing = target_collection.get(include=[])
    existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if existing_ids:
        return 0

    where: Dict[str, Any]
    if len(source_filters) == 1:
        where = {"source": source_filters[0]}
    else:
        where = {"source": {"$in": source_filters}}

    rows = collection.get(where=where, include=["documents", "metadatas"])
    ids = rows.get("ids", []) if isinstance(rows, dict) else []
    documents = rows.get("documents", []) if isinstance(rows, dict) else []
    metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []

    if not ids:
        return 0

    rewritten_ids = [f"{safe_chat_id}:{doc_id}" for doc_id in ids]
    rewritten_metas: List[Dict[str, Any]] = []
    for md in metadatas:
        current = md if isinstance(md, dict) else {}
        copy_md = dict(current)
        copy_md["chat_id"] = safe_chat_id
        copy_md["migrated_from_legacy"] = True
        rewritten_metas.append(copy_md)

    docs_text = [str(doc) for doc in documents]
    embeddings = ollama_embed(docs_text)

    add_documents_in_batches(
        ids=rewritten_ids,
        documents=docs_text,
        embeddings=embeddings,
        metadatas=rewritten_metas,
        collection_obj=target_collection,
    )
    logger.info(
        "[isolation] hydrated chat collection chat_id=%s chunks=%s sources=%s",
        safe_chat_id,
        len(rewritten_ids),
        source_filters,
    )
    return len(rewritten_ids)

def index_uploaded_document(
    job_id: str,
    stored_path: Path,
    safe_name: str,
    file_hash: str,
    file_size_bytes: int,
    index_reset_message: Optional[str],
    chat_id: Optional[str] = None,
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

        avg_chunk_size = int(sum(len(chunk) for chunk in chunks) / max(1, len(chunks)))
        logger.info("[chunking] Indexed %s chunks, avg size: %s chars", len(chunks), avg_chunk_size)

        target_chat_id = normalize_chat_id(chat_id)
        target_collection = get_chat_collection(target_chat_id)

        delete_docs_for_file_hash(file_hash, collection_obj=target_collection)

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
                    "chat_id": target_chat_id,
                }
            )

        set_upload_job(job_id, status="running", message="Writing vectors to index...")
        reset_performed = bool(index_reset_message)
        try:
            add_documents_in_batches(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_obj=target_collection,
            )
        except InvalidArgumentError as exc:
            if not is_dimension_mismatch_error(exc):
                raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

            recreate_collection(str(getattr(target_collection, "name", COLLECTION_NAME)))
            target_collection = get_chat_collection(target_chat_id)
            add_documents_in_batches(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                collection_obj=target_collection,
            )
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
                "chat_id": target_chat_id or None,
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
    collection_obj=None,
) -> None:
    col = collection_obj or collection
    for start in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
        end = start + CHROMA_ADD_BATCH_SIZE
        col.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    with bm25_cache_lock:
        bm25_cache.pop(str(getattr(col, "name", "")), None)


def _split_preserving_blocks(text: str) -> List[Tuple[str, str]]:
    """Return ordered blocks tagged as code|table|list|text, preserving structured content."""
    lines = (text or "").splitlines()
    blocks: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            blocks.append(("code", "\n".join(code_lines).strip()))
            continue

        if "|" in line and line.count("|") >= 2:
            table_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if "|" in next_line and next_line.count("|") >= 2:
                    table_lines.append(next_line)
                    i += 1
                    continue
                break
            blocks.append(("table", "\n".join(table_lines).strip()))
            continue

        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
            list_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", next_line) or re.match(r"^\s{2,}\S", next_line):
                    list_lines.append(next_line)
                    i += 1
                    continue
                break
            blocks.append(("list", "\n".join(list_lines).strip()))
            continue

        text_lines = [line]
        i += 1
        while i < len(lines):
            probe = lines[i]
            probe_stripped = probe.strip()
            if probe_stripped.startswith("```"):
                break
            if "|" in probe and probe.count("|") >= 2:
                break
            if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", probe):
                break
            text_lines.append(probe)
            i += 1
        blocks.append(("text", "\n".join(text_lines).strip()))

    return [(kind, block) for kind, block in blocks if block]


def _sentence_split(text: str) -> List[str]:
    # Lightweight sentence splitter to avoid runtime NLTK downloads in container startup.
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunk(text: str, max_chars: int = MAX_CHUNK_SIZE, overlap_sentences: int = 2) -> List[str]:
    if not (text or "").strip():
        return []

    blocks = _split_preserving_blocks(text)
    chunks: List[str] = []
    previous_tail: List[str] = []

    def flush_current(sentences: List[str]) -> None:
        if not sentences:
            return
        chunk = " ".join(sentences).strip()
        if chunk:
            chunks.append(chunk)

    for block_kind, block_text in blocks:
        if block_kind in {"code", "table", "list"}:
            # Keep structure-heavy blocks intact where possible.
            if len(block_text) <= max_chars:
                if previous_tail:
                    prefix = " ".join(previous_tail).strip()
                    merged = f"{prefix}\n{block_text}".strip()
                    if len(merged) <= max_chars:
                        chunks.append(merged)
                        continue
                chunks.append(block_text)
                continue

            # Very large structural blocks are split by lines as last resort.
            lines = [ln for ln in block_text.splitlines() if ln.strip()]
            current_lines: List[str] = []
            current_len = 0
            for line in lines:
                if current_lines and current_len + len(line) + 1 > max_chars:
                    chunks.append("\n".join(current_lines).strip())
                    current_lines = [line]
                    current_len = len(line)
                else:
                    current_lines.append(line)
                    current_len += len(line) + 1
            if current_lines:
                chunks.append("\n".join(current_lines).strip())
            continue

        sentences = _sentence_split(block_text)
        if not sentences:
            continue

        current: List[str] = previous_tail.copy() if previous_tail else []
        current_len = len(" ".join(current)) if current else 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current and current_len + sent_len + 1 > max_chars:
                flush_current(current)
                current = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current_len = len(" ".join(current)) if current else 0

            if sent_len > max_chars:
                # Worst-case fallback for overlong sentence.
                if current:
                    flush_current(current)
                    current = []
                    current_len = 0
                start = 0
                while start < sent_len:
                    end = min(start + max_chars, sent_len)
                    part = sentence[start:end].strip()
                    if part:
                        chunks.append(part)
                    start = end
                continue

            current.append(sentence)
            current_len += sent_len + 1

        flush_current(current)
        previous_tail = sentences[-overlap_sentences:] if overlap_sentences > 0 else []

    return [chunk for chunk in chunks if chunk.strip()]


def build_parent_child_chunks(text: str) -> List[Dict[str, Any]]:
    """Create larger parent chunks for coherence and smaller child chunks for retrieval."""
    parent_chunks = semantic_chunk(text, max_chars=PARENT_CHUNK_SIZE, overlap_sentences=2)
    if not parent_chunks:
        return []

    records: List[Dict[str, Any]] = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        child_chunks = semantic_chunk(parent_text, max_chars=MAX_CHUNK_SIZE, overlap_sentences=2)
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


def _run_background_maintenance() -> None:
    try:
        start = time.perf_counter()
        migrate_legacy_collection_to_existing_sessions()
        cleanup_orphaned_collections()
        logger.info("[startup] background maintenance finished in %d ms", int((time.perf_counter() - start) * 1000))
    except Exception as exc:
        logger.exception("[startup] background maintenance failed: %s", exc)


@app.on_event("startup")
def startup_background_maintenance() -> None:
    threading.Thread(target=_run_background_maintenance, daemon=True).start()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_context(
    retrieved_items: List[Tuple[str, Optional[dict], float]],
    collection_obj,
    max_chars: int = MAX_CONTEXT_CHARS,
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
                parent_rows = collection_obj.get(
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


def gemini_chat(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List["ChatTurn"]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Tuple[str, int]:
    active_profile = profile or API_GENERATION_PROFILE
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    system_instruction = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )
    user_text = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    contents = []
    if recent_turns:
        for turn in recent_turns[-active_profile.history_turns:]:
            if turn.question.strip():
                contents.append({"role": "user", "parts": [{"text": turn.question.strip()[:1200]}]})
            if turn.answer.strip():
                contents.append({"role": "model", "parts": [{"text": turn.answer.strip()[:1600]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {"temperature": CHAT_TEMPERATURE, "maxOutputTokens": external_max_tokens(active_profile)},
    }

    model_start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
        )
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="Gemini request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini API: {exc}") from exc

    model_response_ms = int((time.perf_counter() - model_start) * 1000)

    if resp.status_code == 400:
        raise HTTPException(status_code=400, detail=f"Gemini API error: {resp.text}")
    if resp.status_code == 401 or resp.status_code == 403:
        raise HTTPException(status_code=401, detail="Gemini API key is invalid or unauthorized.")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Gemini chat failed: {resp.text}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return "No answer generated.", model_response_ms
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    answer = "".join(p.get("text", "") for p in parts).strip() or "No answer generated."
    return answer, model_response_ms


def claude_chat(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List["ChatTurn"]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Tuple[str, int]:
    active_profile = profile or API_GENERATION_PROFILE
    url = "https://api.anthropic.com/v1/messages"
    system_prompt = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )
    user_text = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    messages = []
    if recent_turns:
        for turn in recent_turns[-active_profile.history_turns:]:
            if turn.question.strip():
                messages.append({"role": "user", "content": turn.question.strip()[:1200]})
            if turn.answer.strip():
                messages.append({"role": "assistant", "content": turn.answer.strip()[:1600]})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model_name,
        "max_tokens": external_max_tokens(active_profile),
        "system": system_prompt,
        "messages": messages,
    }

    model_start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
        )
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="Claude request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Anthropic API: {exc}") from exc

    model_response_ms = int((time.perf_counter() - model_start) * 1000)

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Claude API key is invalid or unauthorized.")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Claude chat failed: {resp.text}")

    data = resp.json()
    content_list = data.get("content", [])
    answer = "".join(block.get("text", "") for block in content_list if block.get("type") == "text").strip()
    return answer or "No answer generated.", model_response_ms


def validate_gemini_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate a Gemini API key by listing models."""
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, None
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return False, data.get("error", {}).get("message") or f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"Network error: {exc}"


def validate_claude_key(api_key: str, model: str) -> Tuple[bool, Optional[str]]:
    """Validate a Claude API key by sending a minimal request."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=10,
        )
        if resp.status_code in (200, 400):
            # 400 can be a content/safety rejection, not a key error
            return True, None
        if resp.status_code == 401:
            return False, "Invalid API key."
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return False, data.get("error", {}).get("message") or f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"Network error: {exc}"


def validate_chatgpt_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate a ChatGPT API key by listing OpenAI models."""
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return True, None
        if resp.status_code in (401, 403):
            return False, "Invalid API key."
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return False, data.get("error", {}).get("message") or f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"Network error: {exc}"


def _unique_nonempty(values: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def fetch_latest_gemini_models(api_key: str) -> Tuple[List[str], Optional[str]]:
    """Fetch latest Gemini chat-capable model IDs from Google API."""
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=15,
        )
    except requests.RequestException as exc:
        return GEMINI_MODELS.copy(), f"Gemini model listing failed: {exc}"

    if resp.status_code != 200:
        try:
            data = resp.json()
            message = data.get("error", {}).get("message")
        except Exception:
            message = None
        return GEMINI_MODELS.copy(), message or f"Gemini model listing failed: HTTP {resp.status_code}"

    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    models_raw = data.get("models", []) if isinstance(data, dict) else []
    names: List[str] = []
    for item in models_raw:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        raw_name = str(item.get("name") or "").strip()
        name = raw_name.replace("models/", "", 1)
        if name:
            names.append(name)

    parsed = _unique_nonempty(names)
    if not parsed:
        return GEMINI_MODELS.copy(), "Gemini API returned no compatible models."
    return parsed, None


def fetch_latest_claude_models(api_key: str) -> Tuple[List[str], Optional[str]]:
    """Fetch latest Claude model IDs from Anthropic API."""
    try:
        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        return CLAUDE_MODELS.copy(), f"Claude model listing failed: {exc}"

    if resp.status_code != 200:
        try:
            data = resp.json()
            message = data.get("error", {}).get("message")
        except Exception:
            message = None
        return CLAUDE_MODELS.copy(), message or f"Claude model listing failed: HTTP {resp.status_code}"

    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    model_rows = data.get("data", []) if isinstance(data, dict) else []
    names = [str(row.get("id") or "") for row in model_rows if isinstance(row, dict)]
    parsed = _unique_nonempty(names)
    if not parsed:
        return CLAUDE_MODELS.copy(), "Claude API returned no compatible models."
    return parsed, None


def _is_openai_chat_candidate(model_id: str) -> bool:
    model = (model_id or "").strip().lower()
    if not model:
        return False
    blocked_tokens = (
        "embedding",
        "moderation",
        "whisper",
        "tts",
        "transcribe",
        "audio-preview",
        "omni-moderation",
    )
    return not any(token in model for token in blocked_tokens)


def fetch_latest_chatgpt_models(api_key: str) -> Tuple[List[str], Optional[str]]:
    """Fetch latest GPT model IDs from OpenAI Models API."""
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        return CHATGPT_MODELS.copy(), f"ChatGPT model listing failed: {exc}"

    if resp.status_code != 200:
        try:
            data = resp.json()
            message = data.get("error", {}).get("message")
        except Exception:
            message = None
        return CHATGPT_MODELS.copy(), message or f"ChatGPT model listing failed: HTTP {resp.status_code}"

    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    model_rows = data.get("data", []) if isinstance(data, dict) else []
    names = [str(row.get("id") or "") for row in model_rows if isinstance(row, dict)]
    unique_names = _unique_nonempty(names)
    parsed = [m for m in unique_names if _is_openai_chat_candidate(m)]
    if not parsed:
        parsed = unique_names
    if not parsed:
        return CHATGPT_MODELS.copy(), "ChatGPT API returned no compatible models."
    return parsed, None


def latest_models_for_provider(provider: str, api_key: str) -> Tuple[List[str], Optional[str]]:
    if provider == "gemini":
        return fetch_latest_gemini_models(api_key)
    if provider == "claude":
        return fetch_latest_claude_models(api_key)
    if provider == "chatgpt":
        return fetch_latest_chatgpt_models(api_key)
    return [], f"Unsupported provider: {provider}"


def chatgpt_chat(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List["ChatTurn"]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Tuple[str, int]:
    active_profile = profile or API_GENERATION_PROFILE
    url = "https://api.openai.com/v1/chat/completions"
    system_prompt = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )
    user_text = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if recent_turns:
        for turn in recent_turns[-active_profile.history_turns:]:
            if turn.question.strip():
                messages.append({"role": "user", "content": turn.question.strip()[:1200]})
            if turn.answer.strip():
                messages.append({"role": "assistant", "content": turn.answer.strip()[:1600]})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": external_max_tokens(active_profile),
    }

    model_start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
        )
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="ChatGPT request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach OpenAI API: {exc}") from exc

    model_response_ms = int((time.perf_counter() - model_start) * 1000)
    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="ChatGPT API key is invalid or unauthorized.")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"ChatGPT chat failed: {resp.text}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return "No answer generated.", model_response_ms
    answer = choices[0].get("message", {}).get("content", "")
    return (answer or "No answer generated."), model_response_ms


def ollama_chat(
    question: str,
    context: str,
    model_name: str,
    recent_turns: Optional[List[ChatTurn]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Tuple[str, int]:
    active_profile = profile or LOCAL_GENERATION_PROFILE
    system_prompt = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )

    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if recent_turns:
        # Include a short rolling memory so follow-up questions in a chat remain coherent.
        for turn in recent_turns[-active_profile.history_turns:]:
            q = turn.question.strip()
            a = turn.answer.strip()
            if q:
                messages.append({"role": "user", "content": q[:1200]})
            if a:
                messages.append({"role": "assistant", "content": a[:1600]})

    messages.append({"role": "user", "content": user_prompt})

    model_start = time.perf_counter()
    payload = {
        "model": model_name,
        "stream": False,
        "keep_alive": CHAT_KEEP_ALIVE,
        "options": {
            "temperature": CHAT_TEMPERATURE,
            "top_p": CHAT_TOP_P,
            "num_predict": active_profile.num_predict,
        },
        "messages": messages,
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


def sse_packet(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_chat_messages(
    context: str,
    question: str,
    recent_turns: Optional[List[ChatTurn]],
    history_turns: int,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
        "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
        "Keep answers concise and include source filenames in brackets when possible."
    )
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if recent_turns:
        for turn in recent_turns[-history_turns:]:
            q = turn.question.strip()
            a = turn.answer.strip()
            if q:
                messages.append({"role": "user", "content": q[:1200]})
            if a:
                messages.append({"role": "assistant", "content": a[:1600]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def ollama_chat_stream(
    question: str,
    context: str,
    model_name: str,
    recent_turns: Optional[List[ChatTurn]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Iterator[str]:
    active_profile = profile or LOCAL_GENERATION_PROFILE
    payload = {
        "model": model_name,
        "stream": True,
        "keep_alive": CHAT_KEEP_ALIVE,
        "options": {
            "temperature": CHAT_TEMPERATURE,
            "top_p": CHAT_TOP_P,
            "num_predict": active_profile.num_predict,
        },
        "messages": build_chat_messages(context, question, recent_turns, active_profile.history_turns),
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Chat generation failed: {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload_line = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = str(payload_line.get("message", {}).get("content", ""))
                if chunk:
                    yield chunk
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="Chat generation timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama chat API: {exc}") from exc


def chatgpt_chat_stream(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List[ChatTurn]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Iterator[str]:
    active_profile = profile or API_GENERATION_PROFILE
    payload = {
        "model": model_name,
        "messages": build_chat_messages(context, question, recent_turns, active_profile.history_turns),
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": external_max_tokens(active_profile),
        "stream": True,
    }

    try:
        with requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
            stream=True,
        ) as resp:
            if resp.status_code in (401, 403):
                raise HTTPException(status_code=401, detail="ChatGPT API key is invalid or unauthorized.")
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"ChatGPT chat failed: {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    break
                try:
                    payload_line = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                choices = payload_line.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                chunk = str(delta.get("content") or "")
                if chunk:
                    yield chunk
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="ChatGPT request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach OpenAI API: {exc}") from exc


def gemini_chat_stream(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List[ChatTurn]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Iterator[str]:
    active_profile = profile or API_GENERATION_PROFILE
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent"
    contents: List[Dict[str, Any]] = []
    if recent_turns:
        for turn in recent_turns[-active_profile.history_turns:]:
            if turn.question.strip():
                contents.append({"role": "user", "parts": [{"text": turn.question.strip()[:1200]}]})
            if turn.answer.strip():
                contents.append({"role": "model", "parts": [{"text": turn.answer.strip()[:1600]}]})

    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        f"Context:\n{context}\n\nQuestion: {question}\n\n"
                        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
                    )
                }
            ],
        }
    )

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a careful RAG assistant. Use ONLY the provided context to answer. "
                        "If the answer is not explicitly present in context, say: "
                        "'I do not know based on the provided documents.' "
                        "Keep answers concise and include source filenames in brackets when possible."
                    )
                }
            ]
        },
        "contents": contents,
        "generationConfig": {"temperature": CHAT_TEMPERATURE, "maxOutputTokens": external_max_tokens(active_profile)},
    }

    try:
        with requests.post(
            url,
            params={"key": api_key, "alt": "sse"},
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
            stream=True,
        ) as resp:
            if resp.status_code == 400:
                raise HTTPException(status_code=400, detail=f"Gemini API error: {resp.text}")
            if resp.status_code in (401, 403):
                raise HTTPException(status_code=401, detail="Gemini API key is invalid or unauthorized.")
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Gemini chat failed: {resp.text}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if not data_line:
                    continue
                try:
                    payload_line = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                candidates = payload_line.get("candidates") or []
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                chunk = "".join(str(part.get("text") or "") for part in parts)
                if chunk:
                    yield chunk
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="Gemini request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini API: {exc}") from exc


def claude_chat_stream(
    question: str,
    context: str,
    model_name: str,
    api_key: str,
    recent_turns: Optional[List[ChatTurn]] = None,
    profile: Optional[GenerationProfile] = None,
) -> Iterator[str]:
    active_profile = profile or API_GENERATION_PROFILE
    messages = []
    if recent_turns:
        for turn in recent_turns[-active_profile.history_turns:]:
            if turn.question.strip():
                messages.append({"role": "user", "content": turn.question.strip()[:1200]})
            if turn.answer.strip():
                messages.append({"role": "assistant", "content": turn.answer.strip()[:1600]})

    user_text = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and concisely. Prefer direct quotes/paraphrases from context and avoid assumptions."
    )
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model_name,
        "max_tokens": external_max_tokens(active_profile),
        "stream": True,
        "system": (
            "You are a careful RAG assistant. Use ONLY the provided context to answer. "
            "If the answer is not explicitly present in context, say: 'I do not know based on the provided documents.' "
            "Keep answers concise and include source filenames in brackets when possible."
        ),
        "messages": messages,
    }

    try:
        with requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
            stream=True,
        ) as resp:
            if resp.status_code == 401:
                raise HTTPException(status_code=401, detail="Claude API key is invalid or unauthorized.")
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Claude chat failed: {resp.text}")

            event_name = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    break
                try:
                    payload_line = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                if event_name == "content_block_delta":
                    delta = payload_line.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        chunk = str(delta.get("text") or "")
                        if chunk:
                            yield chunk
    except requests.ReadTimeout as exc:
        raise HTTPException(status_code=504, detail="Claude request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Anthropic API: {exc}") from exc


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
        "chat_model_options": CHAT_MODEL_OPTIONS,
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


@app.get("/models")
def list_chat_models() -> dict:
    ollama_ok, available_models, _ = get_ollama_status()
    available_set = set(available_models)
    ollama_items = [
        {"name": name, "installed": name in available_set, "provider": "local"}
        for name in CHAT_MODEL_OPTIONS
    ]
    gemini_items = [{"name": m, "installed": True, "provider": "gemini"} for m in GEMINI_MODELS]
    claude_items = [{"name": m, "installed": True, "provider": "claude"} for m in CLAUDE_MODELS]
    chatgpt_items = [{"name": m, "installed": True, "provider": "chatgpt"} for m in CHATGPT_MODELS]
    return {
        "default_model": MODEL_NAME,
        "models": ollama_items,
        "ollama_reachable": ollama_ok,
        "external_providers": {
            "gemini": {"models": gemini_items},
            "claude": {"models": claude_items},
            "chatgpt": {"models": chatgpt_items},
        },
    }


@app.post("/api-key/validate")
def validate_api_key_endpoint(req: ApiKeyValidateRequest) -> dict:
    provider = req.provider.strip().lower()
    api_key = (req.api_key or "").strip()
    model = (req.model or "").strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required.")
    if provider not in ("gemini", "claude", "chatgpt"):
        raise HTTPException(status_code=400, detail="provider must be 'gemini', 'claude', or 'chatgpt'.")

    if provider == "gemini":
        valid, error = validate_gemini_key(api_key)
    elif provider == "claude":
        if not model:
            model = CLAUDE_MODELS[0]
        valid, error = validate_claude_key(api_key, model)
    else:
        valid, error = validate_chatgpt_key(api_key)

    if valid:
        latest_models, refresh_error = latest_models_for_provider(provider, api_key)
        model_items = [{"name": m, "installed": True, "provider": provider} for m in latest_models]
        payload: Dict[str, Any] = {"valid": True, "models": model_items}
        if refresh_error:
            payload["model_refresh_error"] = refresh_error
        return payload
    return {"valid": False, "error": error or "API key validation failed."}


@app.get("/documents/sources")
def get_document_sources() -> dict:
    return {"sources": list_indexed_sources()}


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


@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str) -> dict:
    safe_chat_id = normalize_chat_id(chat_id)
    if not safe_chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required.")

    with history_lock:
        with get_history_connection() as conn:
            try:
                conn.execute("BEGIN")
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND role = 'assistant'",
                    (safe_chat_id,),
                ).fetchone()
                deleted_turns = int(row["c"]) if row else 0
                conn.execute("DELETE FROM messages WHERE session_id = ?", (safe_chat_id,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (safe_chat_id,))
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=500, detail=f"Failed to delete chat history: {exc}") from exc

    cname = chat_collection_name(safe_chat_id)
    try:
        client.delete_collection(name=cname)
    except Exception:
        pass

    with bm25_cache_lock:
        bm25_cache.pop(cname, None)

    return {"message": "Chat deleted.", "chat_id": safe_chat_id, "deleted_turns": deleted_turns}


@app.patch("/chat/{chat_id}/title")
def rename_chat(chat_id: str, req: ChatRenameRequest) -> dict:
    safe_chat_id = normalize_chat_id(chat_id)
    safe_title = str(req.title or "").strip()
    if not safe_chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required.")
    if not safe_title:
        raise HTTPException(status_code=400, detail="title is required.")

    final_title = safe_title[:120]
    with history_lock:
        with get_history_connection() as conn:
            try:
                conn.execute("BEGIN")
                row = conn.execute("SELECT id FROM sessions WHERE id = ?", (safe_chat_id,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    raise HTTPException(status_code=404, detail="Chat not found.")
                now = now_iso()
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (final_title, now, safe_chat_id),
                )
                conn.execute("COMMIT")
                logger.info("[chat] renamed chat_id=%s title=%s", safe_chat_id, final_title)
                return {"message": "Chat renamed.", "chat_id": safe_chat_id, "title": final_title}
            except HTTPException:
                raise
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise HTTPException(status_code=500, detail=f"Failed to rename chat: {exc}") from exc


@app.post("/upload")
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), chat_id: Optional[str] = Form(None)) -> dict:
    index_reset_message = ensure_collection_compatible()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    stored_path = UPLOAD_DIR / f"{file_id}_{safe_name}"
    file_size_bytes = 0

    try:
        file_size_bytes, file_hash = save_upload_file(file, stored_path)

        target_chat_id = normalize_chat_id(chat_id)
        target_collection = get_chat_collection(target_chat_id)
        already_indexed, existing_chunks = lookup_existing_file_index(file_hash, collection_obj=target_collection)
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
                "chat_id": target_chat_id or None,
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
        normalize_chat_id(chat_id),
    )

    return {
        "message": "Upload received. Indexing started in background.",
        "file": safe_name,
        "file_size_bytes": file_size_bytes,
        "job_id": job_id,
        "status": "queued",
        "reused_existing_index": False,
        "index_reset_message": index_reset_message,
        "chat_id": normalize_chat_id(chat_id) or None,
    }


@app.get("/upload/jobs/{job_id}")
def upload_job_status(job_id: str) -> dict:
    job = get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return job


def parse_source_filters(raw_filters: Optional[List[str]]) -> List[str]:
    source_filters: List[str] = []
    if isinstance(raw_filters, list):
        for raw in raw_filters:
            value = str(raw).strip()
            if value and value not in source_filters:
                source_filters.append(value)
    return source_filters


def build_source_where_clause(source_filters: List[str]) -> Optional[Dict[str, Any]]:
    if not source_filters:
        return None
    if len(source_filters) == 1:
        return {"source": source_filters[0]}
    return {"source": {"$in": source_filters}}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    request_start = time.perf_counter()
    index_reset_message = ensure_collection_compatible()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    if index_reset_message:
        raise HTTPException(status_code=400, detail=index_reset_message)

    selected_model = (req.model or "").strip()
    provider = (req.provider or "").strip().lower()

    # Normalize 'local' → 'ollama' for internal routing
    if provider == "local":
        provider = "ollama"

    VALID_PROVIDERS = ("ollama", "chatgpt", "gemini", "claude")
    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {req.provider!r}. Must be one of: local, chatgpt, gemini, claude",
        )
    if not selected_model:
        raise HTTPException(status_code=400, detail=f"No model specified for provider '{provider}'")
    if provider in ("gemini", "claude", "chatgpt") and not (req.api_key or "").strip():
        raise HTTPException(status_code=400, detail=f"API key required for {provider}")

    logger.info("[chat/stream] provider=%s model=%s session=%s", provider, selected_model, req.chat_id)
    generation_profile = resolve_generation_profile(provider, selected_model)

    active_chat_id = normalize_chat_id(req.chat_id) or str(uuid.uuid4())
    chat_col = get_chat_collection(active_chat_id)

    source_filters = parse_source_filters(req.source_filters)
    if source_filters:
        hydrate_chat_collection_from_legacy(active_chat_id, source_filters)

    total_docs = chat_col.count()
    if total_docs == 0:
        raise HTTPException(status_code=400, detail="No documents indexed for this chat yet. Upload files first.")

    query_where = build_source_where_clause(source_filters)
    if query_where:
        filtered = chat_col.get(where=query_where, include=[])
        filtered_ids = filtered.get("ids", []) if isinstance(filtered, dict) else []
        if not filtered_ids:
            raise HTTPException(
                status_code=400,
                detail="Selected document source is not indexed yet for this chat. Upload and index the file first.",
            )

    q_embedding = ollama_embed([req.question])[0]
    retrieval_k = get_adaptive_top_k(req.question)

    try:
        retrieved_items = run_hybrid_retrieval(
            col=chat_col,
            query=req.question,
            query_embedding=q_embedding,
            top_k=retrieval_k,
            query_where=query_where,
        )
    except InvalidArgumentError as exc:
        if is_dimension_mismatch_error(exc):
            recreate_collection(str(getattr(chat_col, "name", COLLECTION_NAME)))
            raise HTTPException(
                status_code=400,
                detail=(
                    "Embedding model dimension changed. Index was reset to match the current model. "
                    "Please upload documents again."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    context, selected_metadatas = build_context(
        retrieved_items,
        collection_obj=chat_col,
        max_chars=generation_profile.max_context_chars,
    )
    recent_turns = get_recent_session_turns(active_chat_id, generation_profile.history_turns)

    sources: List[str] = []
    for m in selected_metadatas:
        src = m.get("source") if isinstance(m, dict) else None
        if src and src not in sources:
            sources.append(src)

    def event_stream() -> Iterator[str]:
        chunks: List[str] = []
        model_start = time.perf_counter()

        try:
            if provider == "gemini":
                chunk_iter = gemini_chat_stream(
                    req.question,
                    context,
                    selected_model,
                    req.api_key.strip(),
                    recent_turns,
                    generation_profile,
                )
            elif provider == "claude":
                chunk_iter = claude_chat_stream(
                    req.question,
                    context,
                    selected_model,
                    req.api_key.strip(),
                    recent_turns,
                    generation_profile,
                )
            elif provider == "chatgpt":
                chunk_iter = chatgpt_chat_stream(
                    req.question,
                    context,
                    selected_model,
                    req.api_key.strip(),
                    recent_turns,
                    generation_profile,
                )
            else:
                chunk_iter = ollama_chat_stream(req.question, context, selected_model, recent_turns, generation_profile)

            for chunk in chunk_iter:
                chunks.append(chunk)
                yield sse_packet({"chunk": chunk, "done": False})

            model_response_ms = int((time.perf_counter() - model_start) * 1000)
            answer = "".join(chunks).strip() or "No answer generated."

            turn = ChatTurn(
                timestamp=now_iso(),
                question=req.question,
                answer=answer,
                sources=sources,
                model=selected_model,
                model_response_ms=model_response_ms,
                total_response_ms=int((time.perf_counter() - request_start) * 1000),
            )
            session_id = append_turn_to_session(active_chat_id, turn, provider=provider, model=selected_model)
            yield sse_packet(
                {
                    "chunk": "",
                    "done": True,
                    "chat_id": session_id,
                    "sources": sources,
                    "model": selected_model,
                    "model_response_ms": model_response_ms,
                    "total_response_ms": turn.total_response_ms,
                }
            )
        except HTTPException as exc:
            partial = "".join(chunks).strip()
            if partial:
                interrupted_answer = partial + "\n\nResponse interrupted"
                turn = ChatTurn(
                    timestamp=now_iso(),
                    question=req.question,
                    answer=interrupted_answer,
                    sources=sources,
                    model=selected_model,
                    model_response_ms=int((time.perf_counter() - model_start) * 1000),
                    total_response_ms=int((time.perf_counter() - request_start) * 1000),
                )
                session_id = append_turn_to_session(active_chat_id, turn, provider=provider, model=selected_model)
                yield sse_packet(
                    {
                        "chunk": "\n\nResponse interrupted",
                        "done": True,
                        "interrupted": True,
                        "error": exc.detail,
                        "chat_id": session_id,
                        "sources": sources,
                        "model": selected_model,
                        "model_response_ms": turn.model_response_ms,
                        "total_response_ms": turn.total_response_ms,
                    }
                )
            else:
                yield sse_packet({"chunk": "", "done": True, "error": exc.detail})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    request_start = time.perf_counter()
    index_reset_message = ensure_collection_compatible()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    if index_reset_message:
        raise HTTPException(status_code=400, detail=index_reset_message)

    selected_model = (req.model or "").strip()
    provider = (req.provider or "").strip().lower()

    if provider == "local":
        provider = "ollama"

    VALID_PROVIDERS = ("ollama", "chatgpt", "gemini", "claude")
    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {req.provider!r}. Must be one of: local, chatgpt, gemini, claude",
        )
    if not selected_model:
        raise HTTPException(status_code=400, detail=f"No model specified for provider '{provider}'")
    if provider in ("gemini", "claude", "chatgpt") and not (req.api_key or "").strip():
        raise HTTPException(status_code=400, detail=f"API key required for {provider}")

    logger.info("[chat] provider=%s model=%s session=%s", provider, selected_model, req.chat_id)
    generation_profile = resolve_generation_profile(provider, selected_model)

    active_chat_id = normalize_chat_id(req.chat_id) or str(uuid.uuid4())
    chat_col = get_chat_collection(active_chat_id)

    source_filters = parse_source_filters(req.source_filters)
    if source_filters:
        hydrate_chat_collection_from_legacy(active_chat_id, source_filters)

    total_docs = chat_col.count()
    if total_docs == 0:
        raise HTTPException(status_code=400, detail="No documents indexed for this chat yet. Upload files first.")

    query_where = build_source_where_clause(source_filters)
    if query_where:
        filtered = chat_col.get(where=query_where, include=[])
        filtered_ids = filtered.get("ids", []) if isinstance(filtered, dict) else []
        if not filtered_ids:
            raise HTTPException(
                status_code=400,
                detail="Selected document source is not indexed yet for this chat. Upload and index the file first.",
            )

    q_embedding = ollama_embed([req.question])[0]
    retrieval_k = get_adaptive_top_k(req.question)

    try:
        retrieved_items = run_hybrid_retrieval(
            col=chat_col,
            query=req.question,
            query_embedding=q_embedding,
            top_k=retrieval_k,
            query_where=query_where,
        )
    except InvalidArgumentError as exc:
        if is_dimension_mismatch_error(exc):
            recreate_collection(str(getattr(chat_col, "name", COLLECTION_NAME)))
            raise HTTPException(
                status_code=400,
                detail=(
                    "Embedding model dimension changed. Index was reset to match the current model. "
                    "Please upload documents again."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    context, selected_metadatas = build_context(
        retrieved_items,
        collection_obj=chat_col,
        max_chars=generation_profile.max_context_chars,
    )
    recent_turns = get_recent_session_turns(active_chat_id, generation_profile.history_turns)

    provider = (req.provider or "").strip().lower()
    if provider == "local":
        provider = "ollama"

    if provider == "gemini":
        answer, model_response_ms = gemini_chat(
            req.question,
            context,
            selected_model,
            req.api_key.strip(),
            recent_turns,
            generation_profile,
        )
    elif provider == "claude":
        answer, model_response_ms = claude_chat(
            req.question,
            context,
            selected_model,
            req.api_key.strip(),
            recent_turns,
            generation_profile,
        )
    elif provider == "chatgpt":
        answer, model_response_ms = chatgpt_chat(
            req.question,
            context,
            selected_model,
            req.api_key.strip(),
            recent_turns,
            generation_profile,
        )
    else:
        answer, model_response_ms = ollama_chat(
            req.question,
            context,
            selected_model,
            recent_turns,
            generation_profile,
        )

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
        model=selected_model,
        model_response_ms=model_response_ms,
        total_response_ms=int((time.perf_counter() - request_start) * 1000),
    )
    session_id = append_turn_to_session(active_chat_id, turn, provider=provider, model=selected_model)
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
    deleted_total = 0

    existing = collection.get(include=[])
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if ids:
        collection.delete(ids=ids)
        deleted_total += len(ids)

    for col_name in list_chat_collection_names():
        try:
            rows = client.get_collection(name=col_name).get(include=[])
            cids = rows.get("ids", []) if isinstance(rows, dict) else []
            if cids:
                client.get_collection(name=col_name).delete(ids=cids)
                deleted_total += len(cids)
            with bm25_cache_lock:
                bm25_cache.pop(col_name, None)
        except Exception:
            continue

    with bm25_cache_lock:
        bm25_cache.pop(COLLECTION_NAME, None)

    return {"message": "All indexed documents deleted.", "deleted": deleted_total}
