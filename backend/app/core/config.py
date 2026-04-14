"""
Configuration settings for the application
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "InnovSense"
    
    # Database
    DATABASE_URL: str = "sqlite:///./innovsense.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # ML Model
    MODEL_PATH: str = "./saved_models/"  # This looks in backend/saved_models/
    DEFAULT_MODEL: str = "LogisticRegression"  # Your best model
    
    # Alert thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    MEDIUM_RISK_THRESHOLD: float = 0.6
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()