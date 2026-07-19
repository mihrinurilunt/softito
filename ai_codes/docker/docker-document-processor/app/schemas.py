from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TopWord(BaseModel):
    word: str
    count: int


class DocumentBase(BaseModel):
    id: int
    filename: str
    word_count: int
    character_count: int
    line_count: int
    top_words: list[TopWord]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(DocumentBase):
    message: str


class DocumentListItem(DocumentBase):
    pass
