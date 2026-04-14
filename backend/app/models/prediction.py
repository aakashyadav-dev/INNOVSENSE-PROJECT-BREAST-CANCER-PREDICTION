"""
Prediction result model
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    # Prediction Results
    recurrence_probability = Column(Float, nullable=False)  # 0-1
    risk_category = Column(String)  # Low, Medium, High
    confidence_score = Column(Float)  # Model confidence
    
    # Model Info
    model_used = Column(String)
    model_version = Column(String)
    
    # SHAP Explanation (stored as JSON)
    shap_values = Column(Text)  # JSON string
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="predictions")