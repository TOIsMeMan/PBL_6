from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class NotificationType(str, Enum):
    like = "like"
    comment = "comment"
    follow = "follow"
    system = "system"


class NotificationResponse(BaseModel):
    id: int
    userId: int
    type: NotificationType
    refId: Optional[int] = None
    createdAt: datetime
    seen: bool
    
    class Config:
        from_attributes = True


class NotificationMarkSeen(BaseModel):
    notification_ids: list[int]
