from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.video import Video
from app.models.comment import Comment, CommentStatus
from app.schemas.comment import CommentCreate, CommentResponse, CommentUpdate
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
def create_comment(
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if video exists
    video = db.query(Video).filter(Video.id == comment_data.videoId).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Create comment
    db_comment = Comment(
        userId=current_user.id,
        videoId=comment_data.videoId,
        content=comment_data.content
    )
    
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    
    return db_comment


@router.get("/video/{video_id}", response_model=List[CommentResponse])
def get_video_comments(
    video_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    comments = db.query(Comment).filter(
        Comment.videoId == video_id,
        Comment.status == CommentStatus.visible
    ).offset(skip).limit(limit).all()
    
    return comments


@router.put("/{comment_id}", response_model=CommentResponse)
def update_comment(
    comment_id: int,
    comment_update: CommentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )
    
    if comment.userId != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this comment"
        )
    
    comment.content = comment_update.content
    db.commit()
    db.refresh(comment)
    
    return comment


@router.delete("/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_comment(
    comment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )
    
    if comment.userId != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this comment"
        )
    
    # Soft delete - hide comment
    comment.status = CommentStatus.hidden
    db.commit()
    
    return None
