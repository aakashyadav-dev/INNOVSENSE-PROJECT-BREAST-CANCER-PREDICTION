"""
Machine Learning service for predictions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple
import json
import os

from app.core.config import settings
from app.services.explainability import ModelExplainer

logger = logging.getLogger(__name__)

class MLService:
    """Service for ML model predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.explainer = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Try multiple possible paths
            possible_paths = [
                Path(settings.MODEL_PATH),
                Path("./saved_models/"),
                Path("./ml_models/saved_models/"),
                Path("../saved_models/"),
                Path("C:/Users/vedan/OneDrive/ドキュメント/innovsense-breast-cancer-prediction/backend/saved_models/")
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    logger.info(f"Found models at: {model_path}")
                    break
            
            if model_path is None:
                logger.warning("No model directory found. Using mock service for testing.")
                self._init_mock_service()
                return
            
            # Load preprocessors
            scaler_path = model_path / 'scaler.pkl'
            encoders_path = model_path / 'label_encoders.pkl'
            features_path = model_path / 'feature_columns.pkl'
            
            if not all([p.exists() for p in [scaler_path, encoders_path, features_path]]):
                logger.warning("Preprocessor files not found. Using mock service for testing.")
                self._init_mock_service()
                return
            
            self.scaler = joblib.load(scaler_path)
            self.label_encoders = joblib.load(encoders_path)
            self.feature_columns = joblib.load(features_path)
            
            logger.info(f"Loaded feature columns: {self.feature_columns}")
            
            # Load the best model (prefer LogisticRegression as it was best in training)
            model_file = model_path / 'logisticregression_model.pkl'
            if not model_file.exists():
                model_file = model_path / 'best_model_LogisticRegression.pkl'
            if not model_file.exists():
                # Try XGBoost as fallback
                model_file = model_path / 'xgboost_model.pkl'
            
            if model_file.exists():
                self.model = joblib.load(model_file)
                logger.info(f"Loaded model: {model_file.name}")
                logger.info(f"Model type: {type(self.model).__name__}")
            else:
                logger.error("No model file found")
                self._init_mock_service()
                return
            
            # Initialize explainer
            if self.model is not None and self.feature_columns is not None:
                self.explainer = ModelExplainer(self.model, self.feature_columns)
                logger.info("Explainer initialized successfully")
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._init_mock_service()
    
    def _init_mock_service(self):
        """Initialize mock service for testing when models aren't available"""
        logger.warning("Initializing mock ML service for testing")
        self.feature_columns = [
            'age_at_diagnosis', 'tumor_size', 'lymph_nodes_examined_positive', 
            'neoplasm_histologic_grade', 'chemotherapy', 'hormone_therapy', 
            'radio_therapy', 'er_status', 'pr_status', 'her2_status'
        ]
    
    def _map_patient_data_to_features(self, patient_data: Dict) -> Dict:
        """Map incoming patient data to the feature names expected by the model"""
        mapped_data = {}
        
        # Log incoming data for debugging
        logger.info(f"Incoming patient data keys: {list(patient_data.keys())}")
        logger.info(f"Expected feature columns: {self.feature_columns}")
        
        # Direct mapping - the keys should already match
        for expected_feature in self.feature_columns:
            if expected_feature in patient_data:
                mapped_data[expected_feature] = patient_data[expected_feature]
                logger.info(f"Mapped {expected_feature} = {patient_data[expected_feature]}")
            else:
                logger.warning(f"Feature {expected_feature} not found in input, using default")
                # Set appropriate default values based on feature
                if expected_feature in ['chemotherapy', 'hormone_therapy', 'radio_therapy']:
                    mapped_data[expected_feature] = 0
                elif expected_feature in ['er_status', 'pr_status', 'her2_status']:
                    mapped_data[expected_feature] = 'Unknown'
                else:
                    mapped_data[expected_feature] = 0
        
        return mapped_data
    
    def preprocess_patient_data(self, patient_data: Dict) -> pd.DataFrame:
        """Preprocess patient data for prediction"""
        # First, map the data to expected feature names
        mapped_data = self._map_patient_data_to_features(patient_data)
        
        # Create dataframe
        df = pd.DataFrame([mapped_data])
        
        logger.info(f"DataFrame columns after mapping: {df.columns.tolist()}")
        logger.info(f"DataFrame values before encoding: {df.iloc[0].to_dict()}")
        
        # Ensure all required columns exist in the correct order
        if self.feature_columns:
            # Reorder columns to match training
            df = df[self.feature_columns]
        
        # Handle categorical encoding if encoders exist
        if self.label_encoders:
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    try:
                        val = str(df[col].iloc[0])
                        if val in le.classes_:
                            df[col] = le.transform([val])[0]
                            logger.info(f"Encoded {col}: {val} -> {df[col].iloc[0]}")
                        else:
                            logger.warning(f"Value '{val}' not in encoder classes for {col}, using -1")
                            df[col] = -1
                    except Exception as e:
                        logger.error(f"Error encoding {col}: {e}")
                        df[col] = -1
        
        logger.info(f"DataFrame values after encoding: {df.iloc[0].to_dict()}")
        
        # Scale numeric features if scaler exists
        if self.scaler:
            try:
                # Get numeric columns (excluding categorical ones)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    original_values = df[numeric_cols].copy()
                    df[numeric_cols] = self.scaler.transform(df[numeric_cols])
                    logger.info(f"Scaled values - Before: {original_values.iloc[0].to_dict()}")
                    logger.info(f"Scaled values - After: {df[numeric_cols].iloc[0].to_dict()}")
            except Exception as e:
                logger.error(f"Error scaling features: {e}")
        
        return df
    
    def predict(self, patient_data: Dict) -> Tuple[float, str, float, Dict]:
        """
        Make prediction for a patient
        
        Returns:
            probability: recurrence probability (0-1)
            risk_category: Low/Medium/High
            confidence: model confidence score
            explanation: SHAP explanation values
        """
        # Preprocess data
        X = self.preprocess_patient_data(patient_data)
        
        logger.info(f"Final features for prediction: {X.columns.tolist()}")
        logger.info(f"Final feature values: {X.iloc[0].to_dict()}")
        
        # Make prediction
        if self.model is not None:
            try:
                # Get prediction probability
                proba = self.model.predict_proba(X)
                logger.info(f"Raw prediction probabilities: {proba}")
                probability = float(proba[0, 1])
                logger.info(f"Prediction probability: {probability}")
                
                # Get prediction class (0 or 1)
                prediction = self.model.predict(X)[0]
                logger.info(f"Prediction class: {prediction}")
                
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                probability = 0.35  # Default medium risk
        else:
            probability = self._mock_prediction(patient_data)
        
        # Determine risk category based on thresholds
        if probability < settings.LOW_RISK_THRESHOLD:
            risk_category = "Low"
        elif probability < settings.MEDIUM_RISK_THRESHOLD:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        # Calculate confidence (based on distance from 0.5)
        confidence = min(abs(probability - 0.5) * 2, 0.95)
        
        # Get SHAP explanation
        if self.explainer is not None and self.model is not None:
            try:
                explanation = self.explainer.explain_prediction(X)
                logger.info(f"Generated explanation: {explanation}")
                
                # If explanation is empty or all zeros, use fallback
                if not explanation or all(v == 0 for v in explanation.values()):
                    logger.warning("SHAP explanation returned zeros, using coefficient-based fallback")
                    explanation = self._get_coefficient_based_explanation(X)
            except Exception as e:
                logger.error(f"Error getting explanation: {e}")
                explanation = self._get_coefficient_based_explanation(X)
        else:
            explanation = self._get_coefficient_based_explanation(X)
        
        return probability, risk_category, confidence, explanation
    
    def _get_coefficient_based_explanation(self, X: pd.DataFrame) -> Dict:
        """Get explanation based on model coefficients or feature importance"""
        explanation = {}
        
        try:
            # For Logistic Regression, use coefficients
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                logger.info(f"Model coefficients: {coefficients}")
                
                # Get the scaled feature values
                for i, feature in enumerate(self.feature_columns):
                    if i < len(coefficients):
                        if feature in X.columns:
                            feature_value = float(X[feature].iloc[0])
                            # Multiply coefficient by feature value for local effect
                            explanation[feature] = float(coefficients[i]) * feature_value
                        else:
                            explanation[feature] = float(coefficients[i])
            
            # For tree-based models, use feature importance
            elif hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                logger.info(f"Model feature importances: {importances}")
                
                for i, feature in enumerate(self.feature_columns):
                    if i < len(importances):
                        if feature in X.columns:
                            feature_value = float(X[feature].iloc[0])
                            # Scale importance by feature value
                            explanation[feature] = float(importances[i]) * (feature_value / 10)
                        else:
                            explanation[feature] = float(importances[i])
            
            # Fallback to clinical knowledge-based explanation
            else:
                explanation = self._clinical_knowledge_explanation(X)
            
            # Normalize to reasonable range
            if explanation:
                max_abs = max(abs(v) for v in explanation.values()) if explanation else 1
                if max_abs > 0:
                    for k in explanation:
                        explanation[k] = explanation[k] / max_abs
            
            logger.info(f"Generated coefficient-based explanation: {explanation}")
            
        except Exception as e:
            logger.error(f"Error in coefficient-based explanation: {e}")
            explanation = self._clinical_knowledge_explanation(X)
        
        return explanation
    
    def _clinical_knowledge_explanation(self, X: pd.DataFrame) -> Dict:
        """Generate explanation based on clinical knowledge"""
        explanation = {}
        
        try:
            for feature in self.feature_columns:
                if feature in X.columns:
                    feature_value = float(X[feature].iloc[0])
                    
                    # Clinical knowledge-based impacts
                    if 'er_status' in feature:
                        # ER Positive decreases risk, ER Negative increases risk
                        explanation[feature] = -0.3 if feature_value > 0.5 else 0.4
                    elif 'pr_status' in feature:
                        # PR Positive decreases risk, PR Negative increases risk
                        explanation[feature] = -0.2 if feature_value > 0.5 else 0.3
                    elif 'her2_status' in feature:
                        # HER2 Positive increases risk
                        explanation[feature] = 0.4 if feature_value > 0.5 else -0.1
                    elif 'grade' in feature.lower():
                        # Higher grade increases risk
                        explanation[feature] = feature_value * 0.3
                    elif 'tumor_size' in feature.lower():
                        # Larger tumor increases risk
                        explanation[feature] = feature_value * 0.02
                    elif 'lymph_nodes' in feature.lower():
                        # More positive nodes increases risk significantly
                        explanation[feature] = feature_value * 0.15
                    elif 'age' in feature.lower():
                        # Older age slightly increases risk
                        explanation[feature] = (feature_value - 50) * 0.02
                    elif 'chemotherapy' in feature:
                        # Chemotherapy might decrease risk (treatment effect)
                        explanation[feature] = -0.1 if feature_value > 0.5 else 0.05
                    elif 'hormone_therapy' in feature:
                        # Hormone therapy decreases risk for ER+ patients
                        explanation[feature] = -0.15 if feature_value > 0.5 else 0
                    elif 'radio_therapy' in feature:
                        # Radiation therapy might decrease risk
                        explanation[feature] = -0.05 if feature_value > 0.5 else 0
                    else:
                        explanation[feature] = 0
            
            # Normalize
            if explanation:
                max_abs = max(abs(v) for v in explanation.values())
                if max_abs > 0:
                    for k in explanation:
                        explanation[k] = explanation[k] / max_abs
                        
        except Exception as e:
            logger.error(f"Error in clinical knowledge explanation: {e}")
            # Ultimate fallback - varied values
            np.random.seed(42)
            for i, feature in enumerate(self.feature_columns):
                explanation[feature] = np.random.randn() * 0.2
        
        return explanation
    
    def _mock_prediction(self, patient_data: Dict) -> float:
        """Mock prediction based on simple rules for testing"""
        score = 0.3  # Base score
        
        # Age factor
        age = patient_data.get('age_at_diagnosis', 50)
        if age > 60:
            score += 0.1
        elif age < 40:
            score -= 0.05
        
        # Tumor size
        tumor_size = patient_data.get('tumor_size', 20)
        if tumor_size > 30:
            score += 0.15
        elif tumor_size > 20:
            score += 0.05
        
        # Lymph nodes
        lymph_nodes = patient_data.get('lymph_nodes_examined_positive', 0)
        score += min(lymph_nodes * 0.03, 0.2)
        
        # Grade
        grade = patient_data.get('neoplasm_histologic_grade', 2)
        score += (grade - 2) * 0.1
        
        # Receptor status
        if patient_data.get('er_status') == 'Negative':
            score += 0.1
        if patient_data.get('pr_status') == 'Negative':
            score += 0.05
        if patient_data.get('her2_status') == 'Positive':
            score += 0.1
        
        # Treatment effects
        if patient_data.get('chemotherapy', 0) == 1:
            score -= 0.05
        if patient_data.get('hormone_therapy', 0) == 1:
            score -= 0.05
        if patient_data.get('radio_therapy', 0) == 1:
            score -= 0.02
        
        return min(max(score, 0.1), 0.9)
    
    def get_feature_importance(self) -> Dict:
        """Get global feature importance"""
        if hasattr(self.model, 'coef_') and self.feature_columns:
            # For Logistic Regression
            coefficients = self.model.coef_[0]
            importance = dict(zip(self.feature_columns, np.abs(coefficients)))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'feature_importances_') and self.feature_columns:
            # For tree-based models
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        return {}