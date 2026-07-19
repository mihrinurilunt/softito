from sqlalchemy.orm import Session

from app import models


def create_document(db: Session, filename: str, file_path: str, analysis: dict) -> models.Document:
    document = models.Document(
        filename=filename,
        file_path=file_path,
        word_count=analysis["word_count"],
        character_count=analysis["character_count"],
        line_count=analysis["line_count"],
        top_words=analysis["top_words"],
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def get_documents(db: Session) -> list[models.Document]:
    return db.query(models.Document).order_by(models.Document.created_at.desc()).all()


def get_document(db: Session, document_id: int) -> models.Document | None:
    return db.query(models.Document).filter(models.Document.id == document_id).first()


def delete_document(db: Session, document: models.Document) -> None:
    db.delete(document)
    db.commit()
