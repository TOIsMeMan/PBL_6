from sqlalchemy import Column, BigInteger, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Follow(Base):
    __tablename__ = "Follows"
    
    followerId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), primary_key=True, index=True)
    followeeId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), primary_key=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    follower = relationship("User", foreign_keys=[followerId], back_populates="following")
    followee = relationship("User", foreign_keys=[followeeId], back_populates="followers")