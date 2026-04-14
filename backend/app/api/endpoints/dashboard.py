"""
Dashboard API endpoints for analytics
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction

router = APIRouter()

# Real model performance from training
MODEL_PERFORMANCE = {
    "LogisticRegression": {
        "accuracy": 0.66,
        "precision": 0.47,
        "recall": 0.52,
        "f1_score": 0.49,
        "roc_auc": 0.66
    },
    "XGBoost": {
        "accuracy": 0.70,
        "precision": 0.55,
        "recall": 0.33,
        "f1_score": 0.41,
        "roc_auc": 0.65
    },
    "RandomForest": {
        "accuracy": 0.72,
        "precision": 0.63,
        "recall": 0.27,
        "f1_score": 0.37,
        "roc_auc": 0.65
    }
}

@router.get("/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get overall dashboard statistics"""
    
    # Total patients
    total_patients = db.query(Patient).count()
    
    # Recent predictions (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)
    recent_predictions = db.query(Prediction).filter(
        Prediction.created_at >= week_ago
    ).count()
    
    # Risk distribution
    risk_counts = db.query(
        Prediction.risk_category,
        func.count(Prediction.id).label('count')
    ).group_by(Prediction.risk_category).all()
    
    risk_distribution = {}
    for risk, count in risk_counts:
        if risk:  # Only include non-null risks
            risk_distribution[risk] = count
    
    # Average confidence
    avg_confidence = db.query(
        func.avg(Prediction.confidence_score)
    ).scalar() or 0
    
    return {
        "total_patients": total_patients,
        "recent_predictions": recent_predictions,
        "risk_distribution": risk_distribution,
        "avg_confidence": round(float(avg_confidence), 3),
        "model_performance": MODEL_PERFORMANCE["LogisticRegression"],  # Best model
        "last_updated": datetime.now().isoformat()
    }

@router.get("/predictions/timeline")
async def get_predictions_timeline(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get prediction timeline for charts"""
    start_date = datetime.now() - timedelta(days=days)
    
    predictions = db.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count'),
        func.avg(Prediction.recurrence_probability).label('avg_risk')
    ).filter(
        Prediction.created_at >= start_date
    ).group_by(
        func.date(Prediction.created_at)
    ).order_by('date').all()
    
    return {
        "dates": [str(p[0]) for p in predictions],
        "counts": [p[1] for p in predictions],
        "avg_risk": [float(p[2]) if p[2] else 0 for p in predictions]
    }

@router.get("/patients/recent")
async def get_recent_patients(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent patients with their latest predictions"""
    
    # Subquery to get latest prediction per patient
    subquery = db.query(
        Prediction.patient_id,
        func.max(Prediction.created_at).label('latest_date')
    ).group_by(Prediction.patient_id).subquery()
    
    recent = db.query(
        Patient,
        Prediction
    ).join(
        subquery,
        Patient.id == subquery.c.patient_id
    ).join(
        Prediction,
        (Prediction.patient_id == Patient.id) &
        (Prediction.created_at == subquery.c.latest_date)
    ).order_by(
        Prediction.created_at.desc()
    ).limit(limit).all()
    
    result = []
    for patient, prediction in recent:
        result.append({
            "patient_id": patient.patient_id,
            "age": patient.age,
            "risk_category": prediction.risk_category,
            "probability": prediction.recurrence_probability,
            "confidence": prediction.confidence_score,
            "prediction_date": prediction.created_at.isoformat() if prediction.created_at else None
        })
    
    return result

@router.get("/risk/summary")
async def get_risk_summary(db: Session = Depends(get_db)):
    """Get summary of risk categories"""
    
    total = db.query(Prediction).count()
    if total == 0:
        return {"message": "No predictions yet"}
    
    risk_counts = db.query(
        Prediction.risk_category,
        func.count(Prediction.id).label('count')
    ).group_by(Prediction.risk_category).all()
    
    summary = {}
    for risk, count in risk_counts:
        if risk:
            summary[risk] = {
                "count": count,
                "percentage": round((count / total) * 100, 1)
            }
    
    return summary

@router.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    return MODEL_PERFORMANCE