from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class MessageStatus(str, Enum):
    delivered = "delivered"
    deleted = "deleted"


class MessageCreate(BaseModel):
    receiverId: int
    content: Optional[str] = None
    mediaUrl: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    senderId: int
    receiverId: int
    content: Optional[str] = None
    mediaUrl: Optional[str] = None
    status: MessageStatus
    createdAt: datetime
    
    class Config:
        from_attributes = True
