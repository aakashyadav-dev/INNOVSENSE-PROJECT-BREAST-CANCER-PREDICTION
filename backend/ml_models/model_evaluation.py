"""
Model evaluation and validation module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import joblib
import json

class ModelEvaluator:
    """Advanced model evaluation and visualization"""
    
    def __init__(self, model_path='./saved_models/'):
        self.model_path = model_path
        self.models = {}
        self.results = None
        
    def load_models(self):
        """Load trained models"""
        import os
        
        model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pkl')]
        for model_file in model_files:
            if 'best_model' in model_file:
                name = 'Best_Model'
            else:
                name = model_file.replace('_model.pkl', '').capitalize()
            
            self.models[name] = joblib.load(f'{self.model_path}/{model_file}')
        
        # Load evaluation results
        with open(f'{self.model_path}/evaluation_results.json', 'r') as f:
            self.results = json.load(f)
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Breast Cancer Recurrence Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.model_path}/roc_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.model_path}/pr_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, model) in zip(axes, self.models.items()):
            y_pred = model.predict(X_test)
            cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{name}\nConfusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, X_test, y_test):
        """Generate comprehensive evaluation report"""
        self.plot_roc_curves(X_test, y_test)
        self.plot_precision_recall_curves(X_test, y_test)
        self.plot_confusion_matrices(X_test, y_test)
        
        # Create summary table
        summary = pd.DataFrame(self.results).T
        summary = summary[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        print(summary.round(4))
        
        # Identify best model for each metric
        print("\n" + "-"*60)
        print("Best Model by Metric:")
        print("-"*60)
        for metric in ['roc_auc', 'recall', 'precision', 'f1_score']:
            best = max(self.results, key=lambda x: self.results[x][metric])
            print(f"{metric.upper()}: {best} ({self.results[best][metric]:.4f})")
        
        return summary