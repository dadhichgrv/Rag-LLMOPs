
import os 
from pathlib import Path
from fastapi import APIRouter, File, UploadFile

from src.app.ingestion.azure_storage import upload_input_document
from src.app.ingestion.ingest_documents import ingest_blob_document
from fastapi import APIRouter, File, HTTPException, UploadFile
from src.app.ingestion.ingest_documents import ingest_document

router = APIRouter()



@router.post("/upload")
def upload_document(file : UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    filename = Path(file.filename).name

    try:
        blob_path = upload_input_document(
            file.file,
            filename,
            file.content_type or "application/pdf",
        )
        ingest_blob_document(blob_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc

    return {
        "message": "Document uploaded and ingested successfully",
        "file_name": filename,
        "blob_path": blob_path,
    }

    
