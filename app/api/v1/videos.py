from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from app.database import get_db
from app.models.user import User
from app.models.video import Video, VideoVisibility
from app.models.like import Like
from app.models.comment import Comment
from app.schemas.video import VideoResponse, VideoUpdate
from app.api.deps import get_current_user
from app.utils.validators import validate_video_file_extension
from app.utils.video_processing import extract_video_info, generate_thumbnail, validate_video_duration
import os
import uuid
from app.core.config import settings

router = APIRouter()


@router.post("/", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    title: str = Form(...),
    description: str = Form(None),
    visibility: VideoVisibility = Form(VideoVisibility.public),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Validate file type using utility function
        if not validate_video_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File type not allowed. Allowed types: .mp4, .avi, .mov, .mkv, .webm"
            )
        
        # Create directories
        video_dir = os.path.join(settings.STATIC_DIR, "videos")
        thumbnail_dir = os.path.join(settings.STATIC_DIR, "thumbnails")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        video_path = os.path.join(video_dir, unique_filename)
        
        # Save video file
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving video file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}"
        )
    
    # Get video duration and generate thumbnail using utility functions
    duration_sec = 0
    thumb_url = None
    
    try:
        # Extract video information
        duration_sec, width, height = extract_video_info(video_path)
        print(f"Video info: duration={duration_sec}s, size={width}x{height}")
        
        # Validate video duration (max 120 seconds = 2 minutes)
        if not validate_video_duration(video_path, max_duration=120):
            os.remove(video_path)
            raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video duration exceeds 2 minutes limit"
            )
        
        # Generate thumbnail using utility function
        thumbnail_filename = generate_thumbnail(video_path, thumbnail_dir, frame_time=0.0)
        if thumbnail_filename:
            thumb_url = f"/static/thumbnails/{thumbnail_filename}"
            print(f"Thumbnail generated: {thumbnail_filename}")
        else:
            print("Warning: Could not generate thumbnail")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Warning: Could not process video metadata: {str(e)}")
        # Continue without thumbnail and duration
    
    # Create video record
    try:
        video_url = f"/static/videos/{unique_filename}"
        db_video = Video(
            ownerId=current_user.id,
            title=title,
            description=description,
            url=video_url,
            thumbUrl=thumb_url,
            durationSec=duration_sec,
            visibility=visibility
        )
        
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
    except Exception as e:
        # Cleanup uploaded file if database fails
        if os.path.exists(video_path):
            os.remove(video_path)
        print(f"Database error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video record: {str(e)}"
        )
    
    # Add counts
    video_response = VideoResponse.from_orm(db_video)
    video_response.likes_count = 0
    video_response.comments_count = 0
    
    return video_response


@router.get("/", response_model=List[VideoResponse])
def list_videos(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    videos = db.query(Video).filter(
        Video.visibility == VideoVisibility.public
    ).order_by(desc(Video.createdAt)).offset(skip).limit(limit).all()
    
    # Add counts for each video
    video_responses = []
    for video in videos:
        likes_count = db.query(func.count(Like.userId)).filter(Like.videoId == video.id).scalar()
        comments_count = db.query(func.count(Comment.id)).filter(Comment.videoId == video.id).scalar()
        
        video_dict = video.__dict__.copy()
        video_dict['likes_count'] = likes_count
        video_dict['comments_count'] = comments_count
        video_responses.append(VideoResponse(**video_dict))
    
    return video_responses


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.visibility == VideoVisibility.deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Add counts
    likes_count = db.query(func.count(Like.userId)).filter(Like.videoId == video.id).scalar()
    comments_count = db.query(func.count(Comment.id)).filter(Comment.videoId == video.id).scalar()
    
    video_dict = video.__dict__.copy()
    video_dict['likes_count'] = likes_count
    video_dict['comments_count'] = comments_count
    
    return VideoResponse(**video_dict)


@router.put("/{video_id}", response_model=VideoResponse)
def update_video(
    video_id: int,
    video_update: VideoUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(Video.id == video_id).first()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.ownerId != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this video"
        )
    
    if video_update.title is not None:
        video.title = video_update.title
    if video_update.description is not None:
        video.description = video_update.description
    if video_update.visibility is not None:
        video.visibility = video_update.visibility
    
    db.commit()
    db.refresh(video)
    
    # Add counts
    likes_count = db.query(func.count(Like.userId)).filter(Like.videoId == video.id).scalar()
    comments_count = db.query(func.count(Comment.id)).filter(Comment.videoId == video.id).scalar()
    
    video_dict = video.__dict__.copy()
    video_dict['likes_count'] = likes_count
    video_dict['comments_count'] = comments_count
    
    return VideoResponse(**video_dict)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(Video.id == video_id).first()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.ownerId != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this video"
        )
    
    # Soft delete - mark as deleted
    video.visibility = VideoVisibility.deleted
    db.commit()
    
    return None


@router.get("/user/{user_id}", response_model=List[VideoResponse])
def get_user_videos(
    user_id: int,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # If requesting own videos, show all except deleted
    if user_id == current_user.id:
        videos = db.query(Video).filter(
            Video.ownerId == user_id,
            Video.visibility != VideoVisibility.deleted
        ).order_by(desc(Video.createdAt)).offset(skip).limit(limit).all()
    else:
        # If requesting other user's videos, show only public
        videos = db.query(Video).filter(
            Video.ownerId == user_id,
            Video.visibility == VideoVisibility.public
        ).order_by(desc(Video.createdAt)).offset(skip).limit(limit).all()
    
    # Add counts
    video_responses = []
    for video in videos:
        likes_count = db.query(func.count(Like.userId)).filter(Like.videoId == video.id).scalar()
        comments_count = db.query(func.count(Comment.id)).filter(Comment.videoId == video.id).scalar()
        
        video_dict = video.__dict__.copy()
        video_dict['likes_count'] = likes_count
        video_dict['comments_count'] = comments_count
        video_responses.append(VideoResponse(**video_dict))
    
    return video_responses
