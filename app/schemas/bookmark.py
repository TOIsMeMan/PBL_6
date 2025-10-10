from pydantic import BaseModel
from datetime import datetime


class BookmarkCreate(BaseModel):
    videoId: int


class BookmarkResponse(BaseModel):
    userId: int
    videoId: int
    createdAt: datetime
    
    class Config:
        from_attributes = True
