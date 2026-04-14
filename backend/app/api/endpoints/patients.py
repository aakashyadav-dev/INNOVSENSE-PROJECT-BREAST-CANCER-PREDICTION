"""
Patients API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app.core.database import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction

router = APIRouter()

class PatientResponse(BaseModel):
    patient_id: str
    age: float
    er_status: str
    pr_status: str
    her2_status: str
    created_at: str

@router.get("/", response_model=List[PatientResponse])
async def get_all_patients(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all patients"""
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return [
        {
            "patient_id": p.patient_id,
            "age": p.age,
            "er_status": p.er_status,
            "pr_status": p.pr_status,
            "her2_status": p.her2_status,
            "created_at": p.created_at.isoformat() if p.created_at else None
        }
        for p in patients
    ]

@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get patient by ID"""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "menopausal_state": patient.menopausal_state,
        "tumor_size": patient.tumor_size,
        "lymph_nodes_positive": patient.lymph_nodes_positive,
        "histologic_grade": patient.histologic_grade,
        "er_status": patient.er_status,
        "pr_status": patient.pr_status,
        "her2_status": patient.her2_status,
        "pam50_subtype": patient.pam50_subtype,
        "chemotherapy": patient.chemotherapy,
        "chemotherapy_type": patient.chemotherapy_type,
        "chemo_cycles": patient.chemo_cycles,
        "hormone_therapy": patient.hormone_therapy,
        "radio_therapy": patient.radio_therapy,
        "surgery_type": patient.surgery_type,
        "created_at": patient.created_at.isoformat() if patient.created_at else None
    }