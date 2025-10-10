from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.database import engine, Base
from app.api.v1 import auth, users, videos, comments, social, messages, reports, notifications
import os

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="A TikTok clone API built with FastAPI"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directories
try:
    os.makedirs(settings.STATIC_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.STATIC_DIR, "videos"), exist_ok=True)
    os.makedirs(os.path.join(settings.STATIC_DIR, "thumbnails"), exist_ok=True)
    os.makedirs(os.path.join(settings.STATIC_DIR, "avatars"), exist_ok=True)
except FileExistsError:
    pass  # Directories already exist

# Mount static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["Users"])
app.include_router(videos.router, prefix=f"{settings.API_V1_STR}/videos", tags=["Videos"])
app.include_router(comments.router, prefix=f"{settings.API_V1_STR}/comments", tags=["Comments"])
app.include_router(social.router, prefix=f"{settings.API_V1_STR}/social", tags=["Social (Likes, Follows, Bookmarks)"])
app.include_router(messages.router, prefix=f"{settings.API_V1_STR}/messages", tags=["Messages"])
app.include_router(reports.router, prefix=f"{settings.API_V1_STR}/reports", tags=["Reports"])
app.include_router(notifications.router, prefix=f"{settings.API_V1_STR}/notifications", tags=["Notifications"])


@app.get("/")
def root():
    return {
        "message": "Welcome to TikTok Clone API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)