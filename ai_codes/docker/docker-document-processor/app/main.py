import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.analyzer import analyze_text
from app.database import Base, engine, get_db


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Docker Document Processor",
    description="A Docker-based FastAPI service that analyzes text files and stores metadata in PostgreSQL.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def health_check():
    return {"message": "Docker Document Processor API is running"}


@app.post("/upload", response_model=schemas.DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported.",
        )

    safe_name = Path(file.filename).name
    stored_name = f"{uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / stored_name

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 text.",
        )
    finally:
        file.file.close()

    analysis = analyze_text(text)
    document = crud.create_document(db, safe_name, str(file_path), analysis)

    return {
        "message": "Document uploaded and analyzed successfully",
        **document.__dict__,
    }


@app.get("/documents", response_model=list[schemas.DocumentListItem])
def list_documents(db: Session = Depends(get_db)):
    return crud.get_documents(db)


@app.get("/documents/{document_id}", response_model=schemas.DocumentListItem)
def get_document(document_id: int, db: Session = Depends(get_db)):
    document = crud.get_document(db, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return document


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: int, db: Session = Depends(get_db)):
    document = crud.get_document(db, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    Path(document.file_path).unlink(missing_ok=True)
    crud.delete_document(db, document)
