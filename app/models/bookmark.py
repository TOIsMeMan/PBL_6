from sqlalchemy import Column, BigInteger, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Bookmark(Base):
    __tablename__ = "Bookmarks"
    
    userId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), primary_key=True)
    videoId = Column(BigInteger, ForeignKey("Videos.id", ondelete="CASCADE"), primary_key=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="bookmarks")
    video = relationship("Video", back_populates="bookmarks")
