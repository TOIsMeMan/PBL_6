from sqlalchemy import Column, BigInteger, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class MessageStatus(str, enum.Enum):
    delivered = "delivered"
    deleted = "deleted"


class Message(Base):
    __tablename__ = "Messages"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    senderId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    receiverId = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text)
    mediaUrl = Column(String(500))
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(Enum(MessageStatus), nullable=False, default=MessageStatus.delivered)
    
    # Relationships
    sender = relationship("User", foreign_keys=[senderId], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiverId], back_populates="received_messages")
