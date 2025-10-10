from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app.models.user import User
from app.models.report import Report, ReportStatus, ReportDecision
from app.schemas.report import ReportCreate, ReportResponse, ReportUpdate
from app.api.deps import get_current_user

router = APIRouter()


@router.post("/", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
def create_report(
    report_data: ReportCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create report
    db_report = Report(
        reporterId=current_user.id,
        targetType=report_data.targetType,
        targetId=report_data.targetId,
        reason=report_data.reason,
        note=report_data.note
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    return db_report


@router.get("/", response_model=List[ReportResponse])
def list_reports(
    skip: int = 0,
    limit: int = 50,
    status: ReportStatus = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Only admins can view all reports
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    query = db.query(Report)
    if status:
        query = query.filter(Report.status == status)
    
    reports = query.offset(skip).limit(limit).all()
    return reports


@router.get("/my", response_model=List[ReportResponse])
def my_reports(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get reports made by current user
    reports = db.query(Report).filter(
        Report.reporterId == current_user.id
    ).offset(skip).limit(limit).all()
    
    return reports


@router.put("/{report_id}", response_model=ReportResponse)
def handle_report(
    report_id: int,
    report_update: ReportUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Only admins can handle reports
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if report_update.status is not None:
        report.status = report_update.status
    if report_update.decision is not None:
        report.decision = report_update.decision
    if report_update.note is not None:
        report.note = report_update.note
    
    if report_update.status == ReportStatus.closed:
        report.handledBy = current_user.id
        report.handledAt = datetime.utcnow()
    
    db.commit()
    db.refresh(report)
    
    return report


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Users can only see their own reports, admins can see all
    if report.reporterId != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this report"
        )
    
    return report
