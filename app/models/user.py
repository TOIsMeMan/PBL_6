from sqlalchemy import Column, BigInteger, String, DateTime, Enum, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base
import enum


class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class UserStatus(str, enum.Enum):
    active = "active"
    blocked = "blocked"


class User(Base):
    __tablename__ = "Users"
    
    id = Column(BigInteger().with_variant(BigInteger, "mysql"), primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    fullName = Column(String(255))
    avatarUrl = Column(String(500))
    role = Column(Enum(UserRole), nullable=False, default=UserRole.user)
    status = Column(Enum(UserStatus), nullable=False, default=UserStatus.active)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    googleSub = Column(String(255), unique=True)
    
    # Relationships
    videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan", foreign_keys="Video.ownerId")
    likes = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    bookmarks = relationship("Bookmark", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    # Following/Followers
    following = relationship(
        "Follow",
        foreign_keys="Follow.followerId",
        back_populates="follower",
        cascade="all, delete-orphan"
    )
    followers = relationship(
        "Follow",
        foreign_keys="Follow.followeeId",
        back_populates="followee",
        cascade="all, delete-orphan"
    )
    
    # Messages
    sent_messages = relationship(
        "Message",
        foreign_keys="Message.senderId",
        back_populates="sender",
        cascade="all, delete-orphan"
    )
    received_messages = relationship(
        "Message",
        foreign_keys="Message.receiverId",
        back_populates="receiver",
        cascade="all, delete-orphan"
    )
    
    # Reports
    reports_made = relationship(
        "Report",
        foreign_keys="Report.reporterId",
        back_populates="reporter",
        cascade="all, delete-orphan"
    )
    reports_handled = relationship(
        "Report",
        foreign_keys="Report.handledBy",
        back_populates="handler"
    )