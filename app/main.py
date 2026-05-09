from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import legacy_app as legacy
from routes import chat_router, documents_router, history_router, models_router

app = FastAPI(title="Phi Mini RAG Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(legacy.BASE_DIR / "static")), name="static")


@app.on_event("startup")
def startup_background_maintenance() -> None:
    """Run maintenance on startup without blocking API readiness."""

    legacy.startup_background_maintenance()


@app.get("/")
def index() -> FileResponse:
    """Serve web UI."""

    return FileResponse(str(legacy.BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health(response: Response) -> dict:
    """Return service and runtime health."""

    return legacy.health(response)


app.include_router(models_router)
app.include_router(documents_router)
app.include_router(history_router)
app.include_router(chat_router)
