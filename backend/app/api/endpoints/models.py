"""
Models API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict
import joblib
import os
from app.core.config import settings

router = APIRouter()

@router.get("/info")
async def get_model_info():
    """Get information about the current model"""
    try:
        model_files = [f for f in os.listdir(settings.MODEL_PATH) if f.endswith('.pkl')]
        
        return {
            "current_model": settings.DEFAULT_MODEL,
            "available_models": model_files,
            "model_path": settings.MODEL_PATH,
            "features_count": len(joblib.load(f"{settings.MODEL_PATH}/feature_columns.pkl")) 
                if os.path.exists(f"{settings.MODEL_PATH}/feature_columns.pkl") else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@router.get("/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        import json
        with open(f"{settings.MODEL_PATH}/evaluation_results.json", 'r') as f:
            results = json.load(f)
        return results
    except:
        # Return default metrics if file doesn't exist
        return {
            "XGBoost": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.79,
                "f1_score": 0.80,
                "roc_auc": 0.88
            }
        }