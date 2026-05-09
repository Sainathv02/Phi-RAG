# Mira — RAG Chatbot

RAG chatbot with a FastAPI backend, ChromaDB vector store, Ollama local inference, and optional external providers (ChatGPT, Gemini, Claude) — all from the same UI.

## What This Project Does

- Multiple chat sessions with per-session document attachments.
- Retrieves answers exclusively from indexed document chunks, with source filenames shown on each answer.
- Streams responses to the UI via Server-Sent Events (SSE).
- Provider and model selection per request — local Ollama or any external provider.
- External provider model list is sorted newest-first in the model picker.
- After entering an API key, the model picker reopens automatically so you can choose a model.
- API keys survive page refresh (stored in `sessionStorage`, cleared when the tab closes — never sent to disk).
- Markdown rendered in chat answers (bold, code blocks, bullet lists, headings).
- Chat history persisted in SQLite with session rename and delete.
- Upload polling has a 5-minute timeout — if indexing stalls, the UI surfaces an error instead of hanging.

## Runtime Architecture

- `ollama` — local model runtime.
- `model-init` — waits for Ollama and pre-pulls chat + embedding models.
- `rag-api` — FastAPI backend + static frontend.
- `ChromaDB` — persistent vector index at `app/data/chroma`.
- `SQLite` — chat sessions at `app/data/chat_history.sqlite3`.

Notes:

- `app/data/chat_history.jsonl` is legacy history input; migrated into SQLite on startup when present.
- Provider and model used for each session are stored in the `sessions` table and restored when you reopen a chat.

## Supported Providers

| Provider | Key |
|---|---|
| `local` | None (Ollama handles everything locally) |
| `chatgpt` | OpenAI API key |
| `gemini` | Google AI API key |
| `claude` | Anthropic API key |

The UI sends `provider` and `model` with every chat request. No silent fallback — if either is missing or invalid the backend returns a clear HTTP 400.

## Model Configuration

Default local chat models (`CHAT_MODEL_OPTIONS`):

- `phi4-mini:3.8b-q4_K_M` (default)
- `qwen2.5:3b-instruct-q4_K_M`
- `qwen2.5:1.5b` (low-CPU option)

Default embedding model:

- `nomic-embed-text`

External provider model lists:
- Static fallback lists are included in backend config.
- After API key validation, the backend fetches the live model list from each provider API and returns it to the UI.
- If live listing fails, fallback lists are used.
- In the UI, external models are sorted newest-first based on a known release order. Models not in the known list appear at the bottom.

## Quick Start

```bash
docker compose up --build
```

Wait for:

```text
Uvicorn running on http://0.0.0.0:8000
```

Then open:

- UI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

> On Windows, prefer `127.0.0.1` over `localhost` if IPv6 resolution causes issues.

## Typical Usage Flow

1. Click **+ New Chat**.
2. Attach one or more documents (PDF, DOCX, TXT, MD).
3. Wait for the indexing status bar to confirm completion.
4. Open the model picker (button top-right, or **⌘K** / **Ctrl+K**).
5. Click a provider on the right — for external providers, enter your API key when prompted.
6. After validation the picker reopens — select the model you want.
7. Ask questions. Answers are drawn only from the attached documents.

Behavior details:

- Attachments are scoped to the active chat session.
- Reopening a session restores its provider and model.
- If a document was already indexed, re-uploading reuses the existing index without re-embedding.
- The API key prompt only appears per tab session (keys are kept in `sessionStorage`). Closing the tab requires re-entry.

## Environment Configuration

Copy `.env.example` to `.env` and adjust as needed.

### Core model settings

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `phi4-mini:3.8b-q4_K_M` | Default chat model |
| `CHAT_MODEL_OPTIONS` | `phi4-mini:3.8b-q4_K_M,qwen2.5:3b-instruct-q4_K_M,qwen2.5:1.5b` | Models shown in picker |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for indexing |

### Generation settings

Three profiles control generation depth. More specific profiles take precedence.

**Shared fallback:**

| Variable | Default | Description |
|---|---|---|
| `CHAT_NUM_PREDICT` | `2048` | Max tokens to generate |
| `MAX_CONTEXT_CHARS` | `3000` | Max chars of document context injected |
| `CHAT_HISTORY_TURNS` | `6` | Past turns included in prompt |

**Local profile** (`provider=local`):

| Variable | Default | Description |
|---|---|---|
| `LOCAL_CHAT_NUM_PREDICT` | `2048` | Max tokens for local Ollama |
| `LOCAL_MAX_CONTEXT_CHARS` | `4000` | Context chars for local |
| `LOCAL_CHAT_HISTORY_TURNS` | `8` | History turns for local |

**API profile** (`provider=chatgpt|gemini|claude`):

| Variable | Default | Description |
|---|---|---|
| `API_CHAT_NUM_PREDICT` | `2048` | Max tokens for external providers |
| `API_MAX_CONTEXT_CHARS` | `12000` | Context chars for external |
| `API_CHAT_HISTORY_TURNS` | `20` | History turns for external |

> `LOCAL_CHAT_NUM_PREDICT` controls local answer length. Higher values mean longer answers but slower generation. For low-power machines lower it (e.g. `512`). The old default of `128` (~90 words) cut answers off mid-sentence — `2048` is the correct default.

### Other settings

| Variable | Default | Description |
|---|---|---|
| `CHAT_REQUEST_TIMEOUT_SEC` | `420` | Max seconds for a full chat response |
| `MAX_CHUNK_SIZE` | `1400` | Document chunk size for indexing |
| `EMBED_BATCH_SIZE` | `32` | Embedding batch size |
| `MAX_UPLOAD_MB` | `100` | Max upload file size |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Service, Ollama, and index health |
| `GET` | `/models` | Local + external model lists used by UI |
| `POST` | `/api-key/validate` | Validates external provider key, returns live model list |
| `GET` | `/documents/sources` | Indexed source filenames |
| `POST` | `/upload` | Upload file and start background indexing job |
| `GET` | `/upload/jobs/{job_id}` | Indexing job status (queued / running / completed / failed) |
| `POST` | `/chat/stream` | Streaming chat response (SSE) |
| `POST` | `/chat` | Non-streaming chat response |
| `GET` | `/chat/history` | List saved chat sessions |
| `PATCH` | `/chat/{chat_id}/title` | Rename a chat session |
| `DELETE` | `/chat/{chat_id}` | Delete one chat session |
| `DELETE` | `/chat/history` | Clear all chat sessions |
| `DELETE` | `/documents` | Clear indexed vectors |

## Chat Request Contract

Both `POST /chat` and `POST /chat/stream` accept:

| Field | Required | Description |
|---|---|---|
| `question` | Yes | User question |
| `chat_id` | No | Session ID (created if omitted) |
| `provider` | Yes | `local`, `chatgpt`, `gemini`, or `claude` |
| `model` | Yes | Exact model ID string |
| `api_key` | For external | API key for the provider |
| `source_filters` | No | List of attachment filenames to restrict retrieval to |

Validation:
- Unknown `provider` → HTTP 400.
- Missing `model` → HTTP 400.
- Missing `api_key` for external provider → HTTP 400.
- No silent fallback to another provider or model.

## Troubleshooting

### UI stuck on "Loading…"

- `model-init` may still be pulling models (can take several minutes on first run).
- `ollama` may not be healthy yet.
- Hard refresh (`Ctrl+F5`) clears stale cached assets.

```bash
docker compose ps
docker compose logs model-init ollama rag-api
```

Verify:
- `http://127.0.0.1:8000/health` returns JSON.
- `http://127.0.0.1:8000/models` loads quickly.

### UI shows "Failed to fetch"

Backend is unreachable or a container failed to start.

```bash
docker compose ps
docker compose logs rag-api
```

### `model-init` container exits

Expected — it exits after pulling models successfully.

### Upload stuck on "Indexing…" longer than 5 minutes

The UI will surface a timeout error after ~5 minutes of polling. The server may still be processing in the background. Check:

```bash
docker compose logs rag-api
```

### Wrong source file appears in answer

- Start a new chat.
- Attach only the target file.
- Ask again and verify the `sources` shown under the answer.

### Slow responses on local models

- Switch to `qwen2.5:1.5b` in the model picker.
- Lower `LOCAL_CHAT_NUM_PREDICT` (e.g. `512`) in `.env`.
- Keep document attachments small and questions focused.

### Embedding model changed or dimension mismatch

Re-upload documents to rebuild the vector index after changing `EMBEDDING_MODEL`.

### API key needed again after refresh

API keys are stored in `sessionStorage` and survive page refreshes within the same tab. They are cleared when the tab is closed. This is intentional — keys are never written to disk.

## Common Commands

Start (or rebuild after code changes):

```bash
docker compose up --build
```

Fresh restart without rebuilding:

```bash
docker compose down
docker compose up
```

Full reset — deletes pulled model cache, vectors, and chat history:

```bash
docker compose down -v
docker compose up
```

Inspect resolved compose config:

```bash
docker compose config
```

View live logs:

```bash
docker compose logs -f rag-api
```

## Supported File Types

| Extension | Notes |
|---|---|
| `.pdf` | Supported |
| `.txt` | Supported |
| `.md` | Supported |
| `.docx` | Supported |
| `.doc` | Not supported |
