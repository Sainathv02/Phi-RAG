# Phi RAG Chatbot (Dockerized)

Local RAG chatbot using FastAPI + ChromaDB + Ollama, with a chat-style UI similar to ChatGPT/Claude.

## What This App Does

- Lets you start multiple chat sessions.
- Lets you attach documents in the composer area (next to typing).
- Keeps attachments scoped to the current chat.
- Retrieves answers from indexed document chunks.
- Shows source filenames used in each answer.
- Lets you choose between multiple local chat models.

## Tech Stack

- `ollama`: model runtime
- `model-init`: pulls required models at startup
- `rag-api`: FastAPI backend + static frontend
- `ChromaDB`: persistent vector index in `app/data/chroma`
- `chat_history.jsonl`: persisted chat sessions in `app/data/chat_history.jsonl`

## Model Options

Default configured options:

- `phi4-mini:3.8b-q4_K_M` (best quality of defaults, higher CPU)
- `qwen2.5:3b-instruct-q4_K_M` (balanced speed/quality)
- `qwen2.5:1.5b` (lowest CPU usage, fastest)

Embedding model:

- `nomic-embed-text`

## Quick Start

1. Start the stack:

```bash
docker compose up --build
```

2. Wait until you see:

```text
Uvicorn running on http://0.0.0.0:8000
```

3. Open:

- UI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Note: On some Windows setups, `localhost` may resolve to IPv6 first. Prefer `127.0.0.1`.

## Daily Usage Flow

1. Click `New Chat`.
2. Click `Attach Document` near the input box.
3. Wait for indexing status to show success.
4. Choose a model from the model dropdown.
5. Ask questions.
6. Create another chat for a clean context.

Behavior details:

- New chat starts with no attachments.
- Attachments are chat-scoped.
- Sources shown under answers should match the attached docs for that chat.

## Configuration

Copy `.env.example` to `.env` and customize as needed.

Important model settings:

- `MODEL_NAME=phi4-mini:3.8b-q4_K_M`
- `CHAT_MODEL_OPTIONS=phi4-mini:3.8b-q4_K_M,qwen2.5:3b-instruct-q4_K_M,qwen2.5:1.5b`
- `EMBEDDING_MODEL=nomic-embed-text`

Performance-related settings:

- `CHAT_NUM_PREDICT=128`
- `CHAT_REQUEST_TIMEOUT_SEC=420`
- `CHAT_HISTORY_TURNS=6`
- `MAX_CHUNK_SIZE=1400`
- `EMBED_BATCH_SIZE=32`
- `MAX_UPLOAD_MB=100`

## API Endpoints

- `GET /health`: service + model/index status
- `GET /models`: allowed chat models and install status
- `GET /documents/sources`: indexed source filenames
- `POST /upload`: upload and start background indexing job
- `GET /upload/jobs/{job_id}`: polling status for indexing job
- `POST /chat`: ask a RAG question
- `GET /chat/history`: list chat sessions
- `DELETE /chat/history`: clear chat sessions
- `DELETE /documents`: clear indexed vectors

## Troubleshooting

### UI shows "Failed to fetch"

Usually means backend is not reachable.

Run:

```bash
docker compose ps
docker compose logs model-init rag-api
```

Then verify:

- `http://127.0.0.1:8000/health` returns JSON

### `model-init` fails with `/bin/sh` command errors

This happens if Compose passes `/bin/sh` to `ollama` instead of overriding entrypoint.
Current `docker-compose.yml` in this repo already includes the correct `entrypoint` for `model-init`.

### Source file shown does not match expected document

- Start a new chat.
- Attach only the intended document.
- Ask again and check `sources`.

### CPU is too high / responses too slow

- Use `qwen2.5:1.5b`.
- Lower `CHAT_NUM_PREDICT`.
- Keep prompts and attachments focused.

### Embedding model changed or dimension mismatch

If index compatibility reset happens, re-upload documents so vectors are rebuilt.

## Common Commands

Fresh restart:

```bash
docker compose down
docker compose up --build
```

Inspect resolved config:

```bash
docker compose config
```

## File Types

- Supported: `.pdf`, `.txt`, `.md`, `.docx`
- Not supported: `.doc`
