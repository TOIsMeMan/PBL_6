from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdate
from app.api.deps import get_current_user
from app.utils.validators import validate_image_file_extension, validate_file_size
import os
from app.core.config import settings

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserResponse)
def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user_update.fullName is not None:
        current_user.fullName = user_update.fullName
    if user_update.avatarUrl is not None:
        current_user.avatarUrl = user_update.avatarUrl
    
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.get("/", response_model=List[UserResponse])
def list_users(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.post("/me/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate file type using utility function
    if not validate_image_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type not allowed. Allowed types: .jpg, .jpeg, .png, .gif, .webp"
        )
    
    # Read file content first to check size
    content = await file.read()
    
    # Validate file size (5MB max for images)
    max_size = getattr(settings, 'MAX_IMAGE_SIZE', 5 * 1024 * 1024)  # 5MB default
    if not validate_file_size(len(content), max_size):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed size of {max_size / (1024 * 1024):.1f}MB"
        )
    
    # Create upload directory if not exists
    upload_dir = os.path.join(settings.STATIC_DIR, "avatars")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(upload_dir, f"{current_user.id}{file_ext}")
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    # Update user avatar URL
    avatar_url = f"/static/avatars/{current_user.id}{file_ext}"
    current_user.avatarUrl = avatar_url
    db.commit()
    
    return {"avatar_url": avatar_url}