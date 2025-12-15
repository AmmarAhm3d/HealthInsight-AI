"""
Visualization Module for Healthcare Data
Creates insights visualizations for healthcare professionals
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class HealthcareVisualizer:
    """Create visualizations for medical data analysis"""
    
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_feature_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot distribution of key medical features
        
        Args:
            df: Patient data DataFrame
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Distribution of Medical Features', fontsize=16, fontweight='bold')
        
        features = [
            'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'cholesterol', 'glucose',
            'bmi', 'exercise_hours', 'body_temperature'
        ]
        
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            ax.hist(df[feature], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(feature.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
            print(f"Saved feature distribution plot to {self.output_dir}/feature_distributions.png")
        
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot correlation matrix of features
        
        Args:
            df: Patient data DataFrame
            save: Whether to save the plot
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, linewidths=1)
        plt.title('Correlation Matrix of Medical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {self.output_dir}/correlation_matrix.png")
        
        plt.close()
    
    def plot_disease_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot disease distribution
        
        Args:
            df: Patient data DataFrame
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        disease_counts = df['disease'].value_counts()
        axes[0].bar(['No Disease', 'Disease'], disease_counts.values, 
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0].set_title('Disease Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Patients')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart
        axes[1].pie(disease_counts.values, labels=['No Disease', 'Disease'],
                   autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        axes[1].set_title('Disease Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/disease_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved disease distribution plot to {self.output_dir}/disease_distribution.png")
        
        plt.close()
    
    def plot_model_performance(self, results: dict, save: bool = True) -> None:
        """
        Plot model performance metrics
        
        Args:
            results: Dictionary with model results
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC-like curve using prediction probabilities
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction distribution
        axes[1, 0].hist(results['y_pred_proba'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy metrics
        accuracy = results['accuracy']
        report = results['classification_report']
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        no_disease_metrics = [report['0']['precision'], report['0']['recall'], report['0']['f1-score']]
        disease_metrics = [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, no_disease_metrics, width, label='No Disease', color='green', alpha=0.7)
        axes[1, 1].bar(x + width/2, disease_metrics, width, label='Disease', color='red', alpha=0.7)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'Model Performance Metrics (Accuracy: {accuracy:.2%})', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/model_performance.png', dpi=300, bbox_inches='tight')
            print(f"Saved model performance plot to {self.output_dir}/model_performance.png")
        
        plt.close()
    
    def plot_training_history(self, history, save: bool = True) -> None:
        """
        Plot training history
        
        Args:
            history: Keras training history
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/training_history.png', dpi=300, bbox_inches='tight')
            print(f"Saved training history plot to {self.output_dir}/training_history.png")
        
        plt.close()
    
    def plot_feature_importance_by_disease(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot feature differences between disease and non-disease groups
        
        Args:
            df: Patient data DataFrame
            save: Whether to save the plot
        """
        features = ['age', 'blood_pressure_systolic', 'cholesterol', 'glucose', 'bmi', 'heart_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Distribution by Disease Status', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            
            df[df['disease'] == 0][feature].hist(ax=ax, alpha=0.6, bins=25, 
                                                  color='green', label='No Disease', edgecolor='black')
            df[df['disease'] == 1][feature].hist(ax=ax, alpha=0.6, bins=25,
                                                  color='red', label='Disease', edgecolor='black')
            
            ax.set_title(feature.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_by_disease.png', dpi=300, bbox_inches='tight')
            print(f"Saved feature by disease plot to {self.output_dir}/feature_by_disease.png")
        
        plt.close()
