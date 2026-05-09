"""Routes for document upload and vector source management."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

import legacy_app as legacy

router = APIRouter()


@router.get("/documents/sources")
def get_document_sources() -> dict:
    """Return indexed source file names."""

    return legacy.get_document_sources()


@router.post("/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
) -> dict:
    """Upload and index a document in background."""

    return legacy.upload_document(background_tasks=background_tasks, file=file, chat_id=chat_id)


@router.get("/upload/jobs/{job_id}")
def upload_job_status(job_id: str) -> dict:
    """Return upload background job status."""

    return legacy.upload_job_status(job_id)


@router.delete("/documents")
def delete_all_documents() -> dict:
    """Delete all indexed vectors across shared and chat-scoped collections."""

    return legacy.delete_all_documents()
