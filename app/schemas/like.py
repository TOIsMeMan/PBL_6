from pydantic import BaseModel
from datetime import datetime


class LikeCreate(BaseModel):
    videoId: int


class LikeResponse(BaseModel):
    userId: int
    videoId: int
    createdAt: datetime
    
    class Config:
        from_attributes = True