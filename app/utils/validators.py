"""
Custom validators for the application
"""
from typing import Optional
import re


def validate_username(username: str) -> bool:
    """
    Validate username format
    - 3-50 characters
    - Only alphanumeric, underscore, dash
    """
    if not username or len(username) < 3 or len(username) > 50:
        return False
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, username))


def validate_video_file_extension(filename: str) -> bool:
    """Validate video file extension"""
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def validate_image_file_extension(filename: str) -> bool:
    """Validate image file extension"""
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def validate_file_size(file_size: int, max_size: int) -> bool:
    """Validate file size"""
    return 0 < file_size <= max_size
