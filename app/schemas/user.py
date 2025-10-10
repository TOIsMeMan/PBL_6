from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    user = "user"
    admin = "admin"


class UserStatus(str, Enum):
    active = "active"
    blocked = "blocked"


class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    fullName: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=72)


class UserUpdate(BaseModel):
    fullName: Optional[str] = None
    avatarUrl: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    fullName: Optional[str] = None
    avatarUrl: Optional[str] = None
    role: UserRole
    status: UserStatus
    createdAt: datetime
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None