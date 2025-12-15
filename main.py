"""
HealthInsight AI: Main Analysis Pipeline
Orchestrates data generation, model training, NLP, and visualization
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import MedicalDataGenerator
from dl_model import DiseasePredictionModel
from nlp_analyzer import MedicalNLPAnalyzer
from bio_analyzer import BiologicalDataAnalyzer
from visualizer import HealthcareVisualizer


class HealthInsightAI:
    """Main pipeline for HealthInsight AI medical data analysis"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_generator = MedicalDataGenerator(n_samples=10000)
        self.model = DiseasePredictionModel()
        self.nlp_analyzer = MedicalNLPAnalyzer()
        self.bio_analyzer = BiologicalDataAnalyzer()
        self.visualizer = HealthcareVisualizer()
        
        self.patient_data = None
        self.model_results = None
        
    def generate_data(self):
        """Generate synthetic medical data"""
        print("\n" + "="*60)
        print("STEP 1: Generating Synthetic Medical Data")
        print("="*60)
        
        self.patient_data = self.data_generator.generate_patient_data()
        
        print(f"Generated {len(self.patient_data)} patient records")
        print(f"\nDataset shape: {self.patient_data.shape}")
        print(f"\nFeatures: {list(self.patient_data.columns)}")
        print(f"\nDisease distribution:")
        print(self.patient_data['disease'].value_counts())
        print(f"Disease prevalence: {self.patient_data['disease'].mean():.2%}")
        
        # Save data
        data_path = os.path.join(self.output_dir, 'patient_data.csv')
        self.patient_data.to_csv(data_path, index=False)
        print(f"\nSaved patient data to: {data_path}")
        
        return self.patient_data
    
    def train_model(self):
        """Train deep learning model for disease prediction"""
        print("\n" + "="*60)
        print("STEP 2: Training Deep Learning Model")
        print("="*60)
        
        # Prepare features and target
        X = self.patient_data.drop('disease', axis=1)
        y = self.patient_data['disease']
        
        print(f"Training with {len(X)} samples and {len(X.columns)} features")
        print("Model architecture: Deep Neural Network with 4 hidden layers")
        print("Training in progress...")
        
        # Train model
        self.model_results = self.model.train(X, y, epochs=100, batch_size=32)
        
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE")
        print("="*60)
        print(f"Test Accuracy: {self.model_results['accuracy']:.4f} ({self.model_results['accuracy']*100:.2f}%)")
        print(f"Training samples: {self.model_results['train_size']}")
        print(f"Test samples: {self.model_results['test_size']}")
        
        print("\nClassification Report:")
        report = self.model_results['classification_report']
        print(f"  Class 0 (No Disease) - Precision: {report['0']['precision']:.4f}, Recall: {report['0']['recall']:.4f}, F1: {report['0']['f1-score']:.4f}")
        print(f"  Class 1 (Disease)    - Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1: {report['1']['f1-score']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.model_results['confusion_matrix'])
        
        # Save model
        self.model.save_model()
        print("\nModel saved to: models/")
        
        return self.model_results
    
    def analyze_medical_text(self):
        """Perform NLP analysis on medical notes"""
        print("\n" + "="*60)
        print("STEP 3: NLP Analysis of Medical Notes")
        print("="*60)
        
        # Generate medical notes
        medical_notes = self.data_generator.generate_medical_notes(n_notes=100)
        
        print(f"Generated {len(medical_notes)} medical notes")
        print(f"\nSample note:")
        print(f"  '{medical_notes[0]}'")
        
        # Analyze notes
        nlp_results = self.nlp_analyzer.analyze_notes(medical_notes)
        
        print(f"\n{'='*60}")
        print("NLP ANALYSIS RESULTS")
        print("="*60)
        print(f"Total notes analyzed: {nlp_results['total_notes']}")
        print(f"Unique symptoms identified: {nlp_results['unique_symptoms']}")
        print(f"Unique conditions identified: {nlp_results['unique_conditions']}")
        print(f"Average note length: {nlp_results['avg_note_length']:.1f} words")
        
        print("\nMost common symptoms:")
        for symptom, count in nlp_results['most_common_symptoms'][:5]:
            print(f"  - {symptom}: {count}")
        
        print("\nMost common conditions:")
        for condition, count in nlp_results['most_common_conditions'][:5]:
            print(f"  - {condition}: {count}")
        
        # Save results
        nlp_path = os.path.join(self.output_dir, 'nlp_analysis.json')
        with open(nlp_path, 'w') as f:
            json.dump(nlp_results, f, indent=2)
        print(f"\nNLP results saved to: {nlp_path}")
        
        return nlp_results
    
    def analyze_biological_data(self):
        """Perform Biopython analysis"""
        print("\n" + "="*60)
        print("STEP 4: Biological Sequence Analysis with Biopython")
        print("="*60)
        
        # Generate and analyze sequences
        sequences = self.bio_analyzer.generate_sample_sequences(n_sequences=20)
        
        print(f"Generated {len(sequences)} DNA sequences")
        print(f"\nSample sequence (first 50 bases):")
        print(f"  {str(sequences[0])[:50]}...")
        
        # Batch analysis
        bio_results = self.bio_analyzer.medical_genomics_summary()
        
        print(f"\n{'='*60}")
        print("BIOLOGICAL SEQUENCE ANALYSIS")
        print("="*60)
        genomic = bio_results['genomic_analysis']
        print(f"Sequences analyzed: {genomic['total_sequences']}")
        print(f"Average sequence length: {genomic['avg_length']:.1f} bp")
        print(f"Length range: {genomic['min_length']} - {genomic['max_length']} bp")
        print(f"Average GC content: {genomic['avg_gc_content']:.2f}%")
        print(f"GC content range: {genomic['min_gc_content']:.2f}% - {genomic['max_gc_content']:.2f}%")
        
        print("\nMedical Relevance:")
        for key, value in bio_results['interpretation'].items():
            print(f"  - {value}")
        
        # Save results
        bio_path = os.path.join(self.output_dir, 'biological_analysis.json')
        with open(bio_path, 'w') as f:
            # Convert to JSON-serializable format
            save_data = {
                'genomic_analysis': genomic,
                'interpretation': bio_results['interpretation']
            }
            json.dump(save_data, f, indent=2)
        print(f"\nBiological analysis results saved to: {bio_path}")
        
        return bio_results
    
    def create_visualizations(self):
        """Create all visualizations"""
        print("\n" + "="*60)
        print("STEP 5: Creating Visualizations for Healthcare Professionals")
        print("="*60)
        
        # Feature distributions
        print("\nGenerating feature distribution plots...")
        self.visualizer.plot_feature_distribution(self.patient_data)
        
        # Correlation matrix
        print("Generating correlation matrix...")
        self.visualizer.plot_correlation_matrix(self.patient_data)
        
        # Disease distribution
        print("Generating disease distribution plots...")
        self.visualizer.plot_disease_distribution(self.patient_data)
        
        # Feature comparison by disease
        print("Generating feature comparison by disease...")
        self.visualizer.plot_feature_importance_by_disease(self.patient_data)
        
        # Model performance
        print("Generating model performance visualizations...")
        self.visualizer.plot_model_performance(self.model_results)
        
        # Training history
        print("Generating training history plots...")
        self.visualizer.plot_training_history(self.model.history)
        
        print(f"\nAll visualizations saved to: visualizations/")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("STEP 6: Generating Comprehensive Report")
        print("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'total_patients': len(self.patient_data),
                'features': len(self.patient_data.columns) - 1,
                'disease_prevalence': float(self.patient_data['disease'].mean())
            },
            'model_performance': {
                'accuracy': float(self.model_results['accuracy']),
                'test_samples': int(self.model_results['test_size']),
                'train_samples': int(self.model_results['train_size']),
                'achieved_target': self.model_results['accuracy'] >= 0.92
            },
            'files_generated': {
                'data': 'results/patient_data.csv',
                'model': 'models/disease_prediction_model.h5',
                'visualizations': 'visualizations/*.png'
            }
        }
        
        report_path = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_path}")
        return report
    
    def run_full_pipeline(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print(" " * 15 + "HEALTHINSIGHT AI")
        print(" " * 10 + "Medical Data Analyzer Pipeline")
        print("="*70)
        
        # Step 1: Generate data
        self.generate_data()
        
        # Step 2: Train model
        self.train_model()
        
        # Step 3: NLP analysis
        self.analyze_medical_text()
        
        # Step 4: Biological analysis
        self.analyze_biological_data()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        # Step 6: Generate report
        report = self.generate_report()
        
        # Final summary
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"\n✓ Generated {report['dataset']['total_patients']} patient records")
        print(f"✓ Trained deep learning model with {report['model_performance']['accuracy']*100:.2f}% accuracy")
        print(f"✓ Target accuracy (92%): {'ACHIEVED ✓' if report['model_performance']['achieved_target'] else 'NOT ACHIEVED'}")
        print(f"✓ Performed NLP analysis on medical notes")
        print(f"✓ Analyzed biological sequences with Biopython")
        print(f"✓ Created comprehensive visualizations for healthcare professionals")
        print(f"\nAll results saved to: {self.output_dir}/")
        print(f"All visualizations saved to: visualizations/")
        print(f"Model saved to: models/")
        
        print("\n" + "="*70)
        print("Ready for iteration and further development!")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = HealthInsightAI()
    pipeline.run_full_pipeline()
