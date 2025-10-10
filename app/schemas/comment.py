from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class CommentStatus(str, Enum):
    visible = "visible"
    hidden = "hidden"


class CommentBase(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)


class CommentCreate(CommentBase):
    videoId: int


class CommentUpdate(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)


class CommentResponse(BaseModel):
    id: int
    userId: int
    videoId: int
    content: str
    status: CommentStatus
    createdAt: datetime
    
    class Config:
        from_attributes = True