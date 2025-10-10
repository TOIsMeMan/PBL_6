from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ReportTargetType(str, Enum):
    video = "video"
    comment = "comment"
    user = "user"


class ReportStatus(str, Enum):
    open = "open"
    closed = "closed"


class ReportDecision(str, Enum):
    hide_video = "hide_video"
    delete_video = "delete_video"
    block_user = "block_user"
    reject = "reject"


class ReportCreate(BaseModel):
    targetType: ReportTargetType
    targetId: int
    reason: str = Field(..., min_length=1, max_length=500)
    note: Optional[str] = Field(None, max_length=1000)


class ReportUpdate(BaseModel):
    status: Optional[ReportStatus] = None
    decision: Optional[ReportDecision] = None
    note: Optional[str] = Field(None, max_length=1000)


class ReportResponse(BaseModel):
    id: int
    reporterId: int
    targetType: ReportTargetType
    targetId: int
    reason: str
    note: Optional[str] = None
    status: ReportStatus
    decision: Optional[ReportDecision] = None
    handledBy: Optional[int] = None
    handledAt: Optional[datetime] = None
    createdAt: datetime
    
    class Config:
        from_attributes = True
