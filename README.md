# Phi RAG Chatbot (Dockerized)

A local Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, ChromaDB, and Ollama.

It supports:
- Uploading `PDF`, `TXT`, `MD`, and `DOCX` documents
- Background indexing for large files (no long request blocking)
- Parent-child chunking for better retrieval precision + answer context quality
- Persistent chat history with chat sessions

## Stack Overview

- `ollama` service: model runtime
- `model-init` service: pulls chat + embedding models at startup
- `rag-api` service: FastAPI API + static UI
- `ChromaDB` (persistent): vector index stored in `app/data/chroma`
- `chat_history.jsonl`: persisted chat sessions in `app/data/chat_history.jsonl`

## Default Models

- Chat model: `phi4-mini`
- Embedding model: `nomic-embed-text`

These defaults are set in both `docker-compose.yml` and `.env.example`.

## Quick Start

1. Start services:

```bash
docker compose up --build
```

2. Wait for initial model pull to finish (first run is slower).

3. Open:
- UI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Note: On some Windows setups, `localhost` resolves to IPv6 first and may not work with the Docker port binding. Use `127.0.0.1`.

## How Upload and Indexing Works

1. File is streamed to disk with size enforcement (`MAX_UPLOAD_MB`).
2. File hash is computed (`sha256`) for duplicate detection.
3. If same file hash is already indexed (same index schema), vectors are reused.
4. Otherwise indexing runs in the background job queue:
- extract text
- build parent-child chunks
- embed child chunks
- write vectors to Chroma in batches
5. UI polls job status until completed/failed.

This avoids request timeouts on larger files.

## Parent-Child Chunking Strategy

- Parent chunk: larger section for coherence (`PARENT_CHUNK_SIZE`)
- Child chunk: smaller chunk for embedding/retrieval (`MAX_CHUNK_SIZE`)
- During answer building, retrieved child hits are expanded to nearby child chunks from the same parent (`PARENT_WINDOW_CHILD_SPAN`)

Why this is used:
- Smaller embedded chunks improve retrieval precision.
- Parent-window expansion gives richer context to the chat model.

## API Endpoints

- `GET /health`
	- Returns health and model/index state
	- Returns `503` when Ollama is unreachable or index reset is required

- `POST /upload`
	- Accepts file upload
	- Returns immediately with job details for new indexing:
		- `job_id`, `status=queued`
	- If file already indexed, returns immediate reuse payload (`reused_existing_index=true`)

- `GET /upload/jobs/{job_id}`
	- Poll upload/indexing job status
	- States: `queued`, `running`, `completed`, `failed`

- `POST /chat`
	- Ask a question against indexed docs
	- Request: `question`, optional `chat_id`, optional `top_k`
	- Response: `answer`, `sources`, `chat_id`, `model_response_ms`, `total_response_ms`

- `GET /chat/history?limit_chats=20&limit_turns=50`
	- Returns chat sessions with turns

- `DELETE /chat/history`
	- Clears all chat sessions

- `DELETE /documents`
	- Clears indexed vectors

## Configuration

Copy `.env.example` to `.env` to override values.

Important settings:

- Models
	- `MODEL_NAME=phi4-mini`
	- `EMBEDDING_MODEL=nomic-embed-text`

- Chunking / retrieval
	- `MAX_CHUNK_SIZE=1400`
	- `CHUNK_OVERLAP=100`
	- `PARENT_CHUNK_SIZE=3600`
	- `PARENT_CHUNK_OVERLAP=500`
	- `PARENT_WINDOW_CHILD_SPAN=2`
	- `DEFAULT_TOP_K=4`
	- `RETRIEVAL_MAX_DISTANCE=1.1`
	- `MIN_RETRIEVAL_RESULTS=2`
	- `DOC_SNIPPET_CHARS=900`

- Embedding performance/safety
	- `EMBED_BATCH_SIZE=32`
	- `EMBED_LEGACY_WORKERS=8`
	- `EMBED_NUM_CTX=512`
	- `MAX_EMBED_NUM_CTX=2048`
	- `EMBED_CHARS_PER_TOKEN=4`
	- `CHROMA_ADD_BATCH_SIZE=256`
	- `INDEX_MAX_CHUNKS=0` (`0` means no cap)

- Chat generation
	- `CHAT_TEMPERATURE=0.1`
	- `CHAT_TOP_P=0.9`
	- `CHAT_NUM_PREDICT=128`
	- `CHAT_KEEP_ALIVE=30m`
	- `CHAT_REQUEST_TIMEOUT_SEC=420`
	- `MAX_CONTEXT_CHARS=3000`

- Limits/history
	- `MAX_UPLOAD_MB=100`
	- `MAX_HISTORY_ITEMS=500`

## Common Operations

Restart with fresh build:

```bash
docker compose down
docker compose up --build
```

Check resolved compose config:

```bash
docker compose config
```

## Troubleshooting

- UI cannot open on `localhost`
	- Use `http://127.0.0.1:8000`.

- Large uploads timeout
	- Upload is now background-indexed. Keep the page open and let job polling finish.
	- If still slow, reduce `MAX_CHUNK_SIZE`, raise `INDEX_MAX_CHUNKS`, or use a smaller/faster embedding model.

- Embedding context errors (`input length exceeds context length`)
	- The app auto-truncates and can auto-increase `num_ctx` up to `MAX_EMBED_NUM_CTX`.
	- Tune `EMBED_CHARS_PER_TOKEN` down (for example `3`) if needed.

- Chat generation timeout on CPU
	- Lower `CHAT_NUM_PREDICT` or increase `CHAT_REQUEST_TIMEOUT_SEC`.

- Embedding model changed
	- Index schema/embedding compatibility checks may trigger reset.
	- Re-upload documents after model changes.

## File Support

- Supported: `.pdf`, `.txt`, `.md`, `.docx`
- Not supported: legacy `.doc`
