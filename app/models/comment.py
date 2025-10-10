from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class CommentStatus(str, enum.Enum):
    visible = "visible"
    hidden = "hidden"


class Comment(Base):
    __tablename__ = "Comments"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    videoId = Column(BigInteger, ForeignKey("Videos.id", ondelete="CASCADE"), nullable=False, index=True)
    userId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    content = Column(String(500), nullable=False)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    status = Column(Enum(CommentStatus), nullable=False, default=CommentStatus.visible)
    
    # Relationships
    user = relationship("User", back_populates="comments")
    video = relationship("Video", back_populates="comments")