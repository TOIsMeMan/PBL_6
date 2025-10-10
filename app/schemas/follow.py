from pydantic import BaseModel
from datetime import datetime


class FollowCreate(BaseModel):
    followeeId: int


class FollowResponse(BaseModel):
    followerId: int
    followeeId: int
    createdAt: datetime
    
    class Config:
        from_attributes = True