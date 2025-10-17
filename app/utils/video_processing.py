"""
Video processing utilities
"""
import cv2
import os
from typing import Tuple, Optional
import uuid


def extract_video_info(video_path: str) -> Tuple[int, int, int]:
    """
    Extract video information
    Returns: (duration_sec, width, height)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return (0, 0, 0)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration_sec = int(frame_count / fps) if fps > 0 else 0
        
        cap.release()
        return (duration_sec, width, height)
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return (0, 0, 0)


def generate_thumbnail(video_path: str, output_dir: str, frame_time: float = 0.0) -> Optional[str]:
    """
    Generate thumbnail from video at specified time
    Returns: thumbnail filename or None
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Set frame position
        if frame_time > 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Generate unique filename
        thumbnail_filename = f"{uuid.uuid4()}.jpg"
        thumbnail_path = os.path.join(output_dir, thumbnail_filename)
        
        # Save thumbnail
        cv2.imwrite(thumbnail_path, frame)
        
        return thumbnail_filename
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return None


def validate_video_duration(video_path: str, max_duration: int = 120) -> bool:
    """
    Validate video duration
    Default max: 120 seconds (2 minutes)
    """
    duration, _, _ = extract_video_info(video_path)
    return 0 < duration <= max_duration


def get_video_codec(video_path: str) -> Optional[str]:
    """Get video codec information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        return codec
    except Exception as e:
        print(f"Error getting video codec: {e}")
        return None
