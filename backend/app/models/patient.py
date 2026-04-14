"""
Patient data model
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=False)
    
    # Personal Information
    age = Column(Float, nullable=False)
    menopausal_state = Column(String)
    
    # Cancer Characteristics
    tumor_size = Column(Float, nullable=False)
    lymph_nodes_positive = Column(Integer, default=0)
    histologic_grade = Column(Integer)  # 1, 2, or 3
    er_status = Column(String)  # Positive/Negative
    pr_status = Column(String)  # Positive/Negative
    her2_status = Column(String)  # Positive/Negative
    pam50_subtype = Column(String)  # LumA, LumB, Her2, Basal, claudin-low
    
    # Treatment Information
    chemotherapy = Column(Boolean, default=False)
    chemotherapy_type = Column(String)
    chemo_cycles = Column(Integer, default=0)
    hormone_therapy = Column(Boolean, default=False)
    radio_therapy = Column(Boolean, default=False)
    surgery_type = Column(String)  # MASTECTOMY or BREAST CONSERVING
    
    # Outcome
    overall_survival_months = Column(Float)
    recurrence = Column(Boolean, default=False)
    death_from_cancer = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="patient")