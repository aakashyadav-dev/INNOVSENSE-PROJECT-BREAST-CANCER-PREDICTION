"""
Prediction API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import json

from app.core.database import get_db
from app.services.ml_service import MLService
from app.models.patient import Patient
from app.models.prediction import Prediction

router = APIRouter()

# Initialize ML service (singleton)
ml_service = MLService()

# Pydantic models - UPDATED to match frontend and model features
class PatientInput(BaseModel):
    """Patient input data for prediction - Matches model features exactly"""
    age_at_diagnosis: float = Field(..., description="Patient age at diagnosis", ge=0, le=120)
    tumor_size: float = Field(..., description="Tumor size in mm", ge=0)
    lymph_nodes_examined_positive: int = Field(0, description="Number of positive lymph nodes", ge=0)
    neoplasm_histologic_grade: int = Field(..., description="Tumor grade (1-3)", ge=1, le=3)
    chemotherapy: int = Field(0, description="Received chemotherapy (0/1)")
    hormone_therapy: int = Field(0, description="Received hormone therapy (0/1)")
    radio_therapy: int = Field(0, description="Received radiation therapy (0/1)")
    er_status: str = Field(..., description="ER status (Positive/Negative)")
    pr_status: str = Field(..., description="PR status (Positive/Negative)")
    her2_status: str = Field(..., description="HER2 status (Positive/Negative)")

class PredictionResponse(BaseModel):
    """Prediction response model"""
    patient_id: str
    recurrence_probability: float
    risk_category: str
    confidence_score: float
    explanation: Dict[str, float]
    timestamp: datetime
    disclaimer: str = "This prediction is for informational purposes only and does not replace professional medical advice."

@router.post("/predict", response_model=PredictionResponse)
async def predict_recurrence(
    patient_data: PatientInput,
    db: Session = Depends(get_db)
):
    """
    Predict breast cancer recurrence risk for a patient
    
    Accepts exactly 10 features:
    - age_at_diagnosis
    - tumor_size
    - lymph_nodes_examined_positive
    - neoplasm_histologic_grade
    - chemotherapy (0/1)
    - hormone_therapy (0/1)
    - radio_therapy (0/1)
    - er_status (Positive/Negative)
    - pr_status (Positive/Negative)
    - her2_status (Positive/Negative)
    """
    try:
        # Generate patient ID
        patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Convert to dict
        data_dict = patient_data.dict()
        
        # Log the incoming data for debugging
        print(f"Received prediction request with data: {data_dict}")
        
        # Make prediction
        probability, risk_category, confidence, explanation = ml_service.predict(data_dict)
        
        # Save patient to database
        patient = Patient(
            patient_id=patient_id,
            age=data_dict['age_at_diagnosis'],  # Map to database field
            tumor_size=data_dict['tumor_size'],
            lymph_nodes_positive=data_dict['lymph_nodes_examined_positive'],  # Map to database field
            histologic_grade=data_dict['neoplasm_histologic_grade'],  # Map to database field
            er_status=data_dict['er_status'],
            pr_status=data_dict['pr_status'],
            her2_status=data_dict['her2_status'],
            chemotherapy=bool(data_dict['chemotherapy']),  # Convert 0/1 to bool for DB
            hormone_therapy=bool(data_dict['hormone_therapy']),  # Convert 0/1 to bool for DB
            radio_therapy=bool(data_dict['radio_therapy']),  # Convert 0/1 to bool for DB
            surgery_type="Unknown",  # Default value
            menopausal_state="Unknown"  # Default value
        )
        db.add(patient)
        db.flush()  # Get patient ID
        
        # Save prediction
        prediction = Prediction(
            patient_id=patient.id,
            recurrence_probability=probability,
            risk_category=risk_category,
            confidence_score=confidence,
            model_used="LogisticRegression",  # Based on your training output
            model_version="1.0.0",
            shap_values=json.dumps(explanation)
        )
        db.add(prediction)
        db.commit()
        
        # Return response
        return PredictionResponse(
            patient_id=patient_id,
            recurrence_probability=probability,
            risk_category=risk_category,
            confidence_score=confidence,
            explanation=explanation,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        db.rollback()
        print(f"Error in prediction: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/history/{patient_id}")
async def get_prediction_history(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get prediction history for a patient"""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    predictions = db.query(Prediction).filter(
        Prediction.patient_id == patient.id
    ).order_by(Prediction.created_at.desc()).all()
    
    return [
        {
            "probability": p.recurrence_probability,
            "risk_category": p.risk_category,
            "confidence": p.confidence_score,
            "date": p.created_at,
            "model": p.model_used
        }
        for p in predictions
    ]

@router.get("/features")
async def get_model_features():
    """Get the list of features expected by the model"""
    return {
        "features": [
            "age_at_diagnosis",
            "tumor_size",
            "lymph_nodes_examined_positive",
            "neoplasm_histologic_grade",
            "chemotherapy",
            "hormone_therapy",
            "radio_therapy",
            "er_status",
            "pr_status",
            "her2_status"
        ],
        "count": 10,
        "description": "These are the exact 10 features the model uses for prediction"
    }