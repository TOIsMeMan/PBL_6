from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class ReportTargetType(str, enum.Enum):
    video = "video"
    comment = "comment"
    user = "user"


class ReportStatus(str, enum.Enum):
    open = "open"
    closed = "closed"


class ReportDecision(str, enum.Enum):
    hide_video = "hide_video"
    delete_video = "delete_video"
    block_user = "block_user"
    reject = "reject"


class Report(Base):
    __tablename__ = "Reports"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    reporterId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    targetType = Column(Enum(ReportTargetType), nullable=False)
    targetId = Column(BigInteger, nullable=False)
    reason = Column(String(500), nullable=False)
    note = Column(String(1000))
    status = Column(Enum(ReportStatus), nullable=False, default=ReportStatus.open)
    decision = Column(Enum(ReportDecision))
    handledBy = Column(BigInteger, ForeignKey("Users.id", ondelete="SET NULL"))
    handledAt = Column(DateTime)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    reporter = relationship("User", foreign_keys=[reporterId], back_populates="reports_made")
    handler = relationship("User", foreign_keys=[handledBy], back_populates="reports_handled")
