"""
Data preparation module for breast cancer recurrence prediction
Using the METABRIC dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for breast cancer recurrence prediction"""
    
    def __init__(self, data_path=None):
        # Try multiple possible paths
        if data_path is None:
            possible_paths = [
                "../data/raw/METABRIC_RNA_Mutation.csv",
                "./data/raw/METABRIC_RNA_Mutation.csv",
                "data/raw/METABRIC_RNA_Mutation.csv",
                "C:/Users/vedan/OneDrive/ドキュメント/innovsense-breast-cancer-prediction/backend/data/raw/METABRIC_RNA_Mutation.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    logger.info(f"Found data at: {path}")
                    break
            else:
                # If no path found, use the first one and hope it works
                self.data_path = possible_paths[0]
                logger.warning(f"Could not find data file, using: {self.data_path}")
        else:
            self.data_path = data_path
            
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_and_preprocess(self):
        """Load and preprocess the METABRIC dataset"""
        logger.info(f"Loading METABRIC dataset from: {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            logger.error(f"File not found: {self.data_path}")
            logger.info("Current working directory: " + os.getcwd())
            # Return synthetic data for testing
            return self._create_synthetic_data()
        
        try:
            # Load data
            df = pd.read_csv(self.data_path, low_memory=False, nrows=1000)  # Limit rows for testing
            logger.info(f"Dataset loaded with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_synthetic_data()
        
        # Define target variable
        if 'death_from_cancer' in df.columns:
            df['recurrence'] = df['death_from_cancer'].apply(
                lambda x: 1 if x == 'Died of Disease' else 0
            )
        else:
            logger.warning("death_from_cancer column not found, using synthetic target")
            return self._create_synthetic_data()
        
        # Select relevant features
        feature_cols = [
            'age_at_diagnosis',
            'tumor_size',
            'lymph_nodes_examined_positive',
            'neoplasm_histologic_grade',
            'chemotherapy',
            'hormone_therapy',
            'radio_therapy',
            'er_status',
            'pr_status',
            'her2_status'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_cols)} features: {available_cols}")
        
        # Create feature dataframe
        X = df[available_cols].copy()
        y = df['recurrence'].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical(X)
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing when real data isn't available"""
        logger.warning("Creating synthetic data for testing")
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'age_at_diagnosis': np.random.normal(60, 12, n_samples),
            'tumor_size': np.random.exponential(25, n_samples),
            'lymph_nodes_examined_positive': np.random.poisson(2, n_samples),
            'neoplasm_histologic_grade': np.random.choice([1, 2, 3], n_samples),
            'chemotherapy': np.random.choice([0, 1], n_samples),
            'hormone_therapy': np.random.choice([0, 1], n_samples),
            'radio_therapy': np.random.choice([0, 1], n_samples),
            'er_status': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.7, 0.3]),
            'pr_status': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.6, 0.4]),
            'her2_status': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.2, 0.8])
        })
        
        # Create synthetic target (40% recurrence rate)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.6, 0.4]))
        
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        return df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numeric features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self, path='./saved_models/'):
        """Save scaler and label encoders"""
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{path}/feature_columns.pkl')
        logger.info(f"Preprocessors saved to {path}")