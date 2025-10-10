from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TikTok Clone API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "mysql+pymysql://root:1234@localhost:3306/pbl6"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production-09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    STATIC_DIR: str = "static"
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_VIDEO_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv"]
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".gif"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()