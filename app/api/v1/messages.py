from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from app.database import get_db
from app.models.user import User
from app.models.message import Message, MessageStatus
from app.schemas.message import MessageCreate, MessageResponse
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def send_message(
    message_data: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if receiver exists
    receiver = db.query(User).filter(User.id == message_data.receiverId).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Receiver not found"
        )
    
    # Can't send message to yourself
    if message_data.receiverId == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot send message to yourself"
        )
    
    # Create message
    db_message = Message(
        senderId=current_user.id,
        receiverId=message_data.receiverId,
        content=message_data.content,
        mediaUrl=message_data.mediaUrl
    )
    
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    return db_message


@router.get("/conversation/{user_id}", response_model=List[MessageResponse])
def get_conversation(
    user_id: int,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get messages between current user and specified user
    messages = db.query(Message).filter(
        and_(
            or_(
                and_(Message.senderId == current_user.id, Message.receiverId == user_id),
                and_(Message.senderId == user_id, Message.receiverId == current_user.id)
            ),
            Message.status == MessageStatus.delivered
        )
    ).order_by(Message.createdAt.desc()).offset(skip).limit(limit).all()
    
    return messages


@router.get("/inbox", response_model=List[MessageResponse])
def get_inbox(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get all messages received by current user
    messages = db.query(Message).filter(
        Message.receiverId == current_user.id,
        Message.status == MessageStatus.delivered
    ).order_by(Message.createdAt.desc()).offset(skip).limit(limit).all()
    
    return messages


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_message(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    message = db.query(Message).filter(Message.id == message_id).first()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # Only sender can delete their message
    if message.senderId != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this message"
        )
    
    # Soft delete
    message.status = MessageStatus.deleted
    db.commit()
    
    return None
