from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.notification import Notification
from app.schemas.notification import NotificationResponse, NotificationMarkSeen
from app.api.deps import get_current_user

router = APIRouter()


@router.get("/", response_model=List[NotificationResponse])
def get_notifications(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notifications = db.query(Notification).filter(
        Notification.userId == current_user.id
    ).order_by(Notification.createdAt.desc()).offset(skip).limit(limit).all()
    
    return notifications


@router.get("/unseen/count")
def get_unseen_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    count = db.query(Notification).filter(
        Notification.userId == current_user.id,
        Notification.seen == False
    ).count()
    
    return {"unseen_count": count}


@router.post("/mark-seen", status_code=status.HTTP_200_OK)
def mark_notifications_seen(
    data: NotificationMarkSeen,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Mark specified notifications as seen
    db.query(Notification).filter(
        Notification.id.in_(data.notification_ids),
        Notification.userId == current_user.id
    ).update({"seen": True}, synchronize_session=False)
    
    db.commit()
    
    return {"message": "Notifications marked as seen"}


@router.post("/mark-all-seen", status_code=status.HTTP_200_OK)
def mark_all_seen(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Mark all notifications as seen for current user
    db.query(Notification).filter(
        Notification.userId == current_user.id,
        Notification.seen == False
    ).update({"seen": True}, synchronize_session=False)
    
    db.commit()
    
    return {"message": "All notifications marked as seen"}


@router.delete("/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notification(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.userId == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    db.delete(notification)
    db.commit()
    
    return None
