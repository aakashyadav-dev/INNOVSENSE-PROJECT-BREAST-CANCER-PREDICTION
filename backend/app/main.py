"""
FastAPI main application for InnovSense Breast Cancer Recurrence Prediction
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import logging

from app.api.endpoints import prediction, patients, models, dashboard
from app.core.config import settings
from app.core.database import engine, Base
from app.models.patient import Patient
from app.models.prediction import Prediction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="InnovSense - Breast Cancer Recurrence Prediction API",
    description="AI-powered healthcare system for predicting breast cancer recurrence after chemotherapy",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["Prediction"])
app.include_router(patients.router, prefix="/api/v1/patients", tags=["Patients"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to InnovSense Breast Cancer Recurrence Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"  # Add actual DB check
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )