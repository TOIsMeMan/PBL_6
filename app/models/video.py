from sqlalchemy import Column, BigInteger, Integer, String, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class VideoVisibility(str, enum.Enum):
    public = "public"
    hidden = "hidden"
    deleted = "deleted"


class Video(Base):
    __tablename__ = "Videos"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    ownerId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(120), nullable=False)
    description = Column(String(2200))
    durationSec = Column(Integer)  # CHECK constraint handled by MySQL
    visibility = Column(Enum(VideoVisibility), nullable=False, default=VideoVisibility.public)
    url = Column(String(500), nullable=False)
    hlsUrl = Column(String(500))
    thumbUrl = Column(String(500))
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="videos", foreign_keys=[ownerId])
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="video", cascade="all, delete-orphan")
    bookmarks = relationship("Bookmark", back_populates="video", cascade="all, delete-orphan")