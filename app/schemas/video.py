from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class VideoVisibility(str, Enum):
    public = "public"
    hidden = "hidden"
    deleted = "deleted"


class VideoBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2200)
    visibility: VideoVisibility = VideoVisibility.public


class VideoCreate(VideoBase):
    pass


class VideoUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2200)
    visibility: Optional[VideoVisibility] = None


class VideoResponse(BaseModel):
    id: int
    ownerId: int
    title: str
    description: Optional[str] = None
    durationSec: Optional[int] = None
    visibility: VideoVisibility
    url: str
    hlsUrl: Optional[str] = None
    thumbUrl: Optional[str] = None
    createdAt: datetime
    likes_count: int = 0
    comments_count: int = 0
    
    class Config:
        from_attributes = True