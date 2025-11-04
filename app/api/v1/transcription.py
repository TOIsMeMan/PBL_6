"""
Video Transcription API Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.video import Video
from app.api.deps import get_current_user
from app.services.transcription import get_transcription_service
from app.core.config import settings
import os
import json

router = APIRouter()


@router.post("/{video_id}/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_video(
    video_id: int,
    use_correction: bool = Form(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Transcribe video thành text với timestamps
    
    - **video_id**: ID của video cần transcribe
    - **use_correction**: Có sử dụng Gemini AI để sửa lỗi không (default: True)
    """
    # Check if video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Check permission (chỉ owner mới có thể transcribe)
    if video.ownerId != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to transcribe this video"
        )
    
    # Get video file path
    video_path = os.path.join(settings.STATIC_DIR, video.url.lstrip('/static/'))
    
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found on server"
        )
    
    try:
        # Get transcription service
        transcription_service = get_transcription_service()
        
        # Transcribe video
        gemini_key = getattr(settings, 'GEMINI_API_KEY', None) if use_correction else None
        result = transcription_service.transcribe_video(
            video_path=video_path,
            use_correction=use_correction,
            gemini_api_key=gemini_key
        )
        
        # Update video record với transcript
        video.transcript = result['transcript']
        video.transcriptTimestamps = json.dumps(result['timestamps'], ensure_ascii=False)
        
        db.commit()
        db.refresh(video)
        
        return {
            "message": "Video transcribed successfully",
            "video_id": video_id,
            "transcript": result['transcript'],
            "timestamps": result['timestamps'],
            "confidence": result['confidence'],
            "duration": result['duration']
        }
        
    except Exception as e:
        db.rollback()
        print(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.get("/{video_id}/transcript", status_code=status.HTTP_200_OK)
def get_video_transcript(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Lấy transcript của video (nếu đã được transcribe)
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if not video.transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video has not been transcribed yet"
        )
    
    timestamps = []
    if video.transcriptTimestamps:
        try:
            timestamps = json.loads(video.transcriptTimestamps)
        except:
            pass
    
    return {
        "video_id": video_id,
        "title": video.title,
        "transcript": video.transcript,
        "timestamps": timestamps
    }
