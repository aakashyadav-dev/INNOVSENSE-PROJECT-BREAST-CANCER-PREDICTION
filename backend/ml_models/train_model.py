"""
Model training module for breast cancer recurrence prediction
Trains XGBoost, Random Forest, and Logistic Regression models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation import DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and evaluates ML models for recurrence prediction"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self):
        """Load and prepare data for training"""
        logger.info("Preparing data for training...")
        X, y = self.preprocessor.load_and_preprocess()
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Recurrence rate: {y.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        logger.info("Training models...")
        
        # XGBoost (primary model)
        logger.info("Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        # Logistic Regression (baseline)
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        lr_model.fit(X_train, y_train)
        self.models['LogisticRegression'] = lr_model
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        logger.info("Evaluating models...")
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_test, y_test, cv=StratifiedKFold(5), 
                scoring='roc_auc'
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"\n{name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model based on ROC-AUC
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.models[self.best_model_name]
        logger.info(f"\nBest Model: {self.best_model_name}")
        
        return self.results
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # XGBoost importance
        if 'XGBoost' in self.models:
            importance = self.models['XGBoost'].feature_importances_
            indices = np.argsort(importance)[::-1][:15]
            
            axes[0].barh(range(len(indices)), importance[indices])
            axes[0].set_yticks(range(len(indices)))
            axes[0].set_yticklabels([feature_names[i] for i in indices])
            axes[0].set_xlabel('Importance')
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].invert_yaxis()
        
        # Random Forest importance
        if 'RandomForest' in self.models:
            importance = self.models['RandomForest'].feature_importances_
            indices = np.argsort(importance)[::-1][:15]
            
            axes[1].barh(range(len(indices)), importance[indices])
            axes[1].set_yticks(range(len(indices)))
            axes[1].set_yticklabels([feature_names[i] for i in indices])
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Random Forest Feature Importance')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_models(self, path='./saved_models/'):
        """Save all trained models and preprocessors"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, f'{path}/best_model_{self.best_model_name}.pkl')
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, f'{path}/{name.lower()}_model.pkl')
        
        # Save evaluation results
        with open(f'{path}/evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save preprocessors
        self.preprocessor.save_preprocessors(path)
        
        logger.info(f"All models and preprocessors saved to {path}")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Plot feature importance
        self.plot_feature_importance(self.preprocessor.feature_columns)
        
        # Save models
        self.save_models()
        
        return self.results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline()
    
    # Print final comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4))