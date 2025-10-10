from sqlalchemy import Column, BigInteger, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class NotificationType(str, enum.Enum):
    like = "like"
    comment = "comment"
    follow = "follow"
    system = "system"


class Notification(Base):
    __tablename__ = "Notifications"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    userId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    type = Column(Enum(NotificationType), nullable=False)
    refId = Column(BigInteger)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    seen = Column(Boolean, nullable=False, default=False)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
