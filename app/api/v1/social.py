from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.video import Video
from app.models.like import Like
from app.models.follow import Follow
from app.models.bookmark import Bookmark
from app.schemas.like import LikeResponse
from app.schemas.follow import FollowResponse
from app.schemas.bookmark import BookmarkResponse
from app.api.deps import get_current_user

router = APIRouter()


# ===== LIKES =====

@router.post("/likes/{video_id}", response_model=LikeResponse, status_code=status.HTTP_201_CREATED)
def like_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Check if already liked
    existing_like = db.query(Like).filter(
        Like.userId == current_user.id,
        Like.videoId == video_id
    ).first()
    
    if existing_like:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already liked this video"
        )
    
    # Create like
    db_like = Like(
        userId=current_user.id,
        videoId=video_id
    )
    
    db.add(db_like)
    db.commit()
    db.refresh(db_like)
    
    return db_like


@router.delete("/likes/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def unlike_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    like = db.query(Like).filter(
        Like.userId == current_user.id,
        Like.videoId == video_id
    ).first()
    
    if not like:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Like not found"
        )
    
    db.delete(like)
    db.commit()
    
    return None


@router.get("/likes/video/{video_id}", response_model=List[LikeResponse])
def get_video_likes(
    video_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    likes = db.query(Like).filter(
        Like.videoId == video_id
    ).offset(skip).limit(limit).all()
    
    return likes


# ===== FOLLOWS =====

@router.post("/follow/{user_id}", response_model=FollowResponse, status_code=status.HTTP_201_CREATED)
def follow_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Can't follow yourself
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot follow yourself"
        )
    
    # Check if user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if already following
    existing_follow = db.query(Follow).filter(
        Follow.followerId == current_user.id,
        Follow.followeeId == user_id
    ).first()
    
    if existing_follow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already following this user"
        )
    
    # Create follow
    db_follow = Follow(
        followerId=current_user.id,
        followeeId=user_id
    )
    
    db.add(db_follow)
    db.commit()
    db.refresh(db_follow)
    
    return db_follow


@router.delete("/unfollow/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def unfollow_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    follow = db.query(Follow).filter(
        Follow.followerId == current_user.id,
        Follow.followeeId == user_id
    ).first()
    
    if not follow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Follow relationship not found"
        )
    
    db.delete(follow)
    db.commit()
    
    return None


@router.get("/followers/{user_id}", response_model=List[FollowResponse])
def get_followers(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    followers = db.query(Follow).filter(
        Follow.followeeId == user_id
    ).offset(skip).limit(limit).all()
    
    return followers


@router.get("/following/{user_id}", response_model=List[FollowResponse])
def get_following(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    following = db.query(Follow).filter(
        Follow.followerId == user_id
    ).offset(skip).limit(limit).all()
    
    return following


# ===== BOOKMARKS =====

@router.post("/bookmarks/{video_id}", response_model=BookmarkResponse, status_code=status.HTTP_201_CREATED)
def bookmark_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Check if already bookmarked
    existing_bookmark = db.query(Bookmark).filter(
        Bookmark.userId == current_user.id,
        Bookmark.videoId == video_id
    ).first()
    
    if existing_bookmark:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already bookmarked this video"
        )
    
    # Create bookmark
    db_bookmark = Bookmark(
        userId=current_user.id,
        videoId=video_id
    )
    
    db.add(db_bookmark)
    db.commit()
    db.refresh(db_bookmark)
    
    return db_bookmark


@router.delete("/bookmarks/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_bookmark(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bookmark = db.query(Bookmark).filter(
        Bookmark.userId == current_user.id,
        Bookmark.videoId == video_id
    ).first()
    
    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found"
        )
    
    db.delete(bookmark)
    db.commit()
    
    return None


@router.get("/bookmarks/my", response_model=List[BookmarkResponse])
def get_my_bookmarks(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    bookmarks = db.query(Bookmark).filter(
        Bookmark.userId == current_user.id
    ).offset(skip).limit(limit).all()
    
    return bookmarks
