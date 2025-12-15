"""
Quick usage examples for HealthInsight AI
Demonstrates individual component usage
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print(" " * 20 + "HealthInsight AI Examples")
print("="*70 + "\n")

# Example 1: Generate and view patient data
print("Example 1: Generate Patient Data")
print("-" * 70)
from data_generator import MedicalDataGenerator

generator = MedicalDataGenerator(n_samples=100)
data = generator.generate_patient_data()
print(f"Generated {len(data)} patient records")
print("\nFirst 3 patients:")
print(data.head(3)[['age', 'blood_pressure_systolic', 'cholesterol', 'glucose', 'disease']])
print(f"\nDisease rate: {data['disease'].mean():.2%}\n")

# Example 2: NLP Analysis
print("\nExample 2: Medical Text NLP Analysis")
print("-" * 70)
from nlp_analyzer import MedicalNLPAnalyzer

nlp = MedicalNLPAnalyzer()
sample_note = "Patient presents with severe chest pain and shortness of breath. History of hypertension and diabetes."

symptoms = nlp.extract_symptoms(sample_note)
conditions = nlp.extract_conditions(sample_note)

print(f"Medical Note: '{sample_note}'")
print(f"Extracted Symptoms: {symptoms}")
print(f"Extracted Conditions: {conditions}")
print(f"Sentiment: {nlp.sentiment_analysis(sample_note)}\n")

# Example 3: Biological Sequence Analysis
print("\nExample 3: DNA Sequence Analysis")
print("-" * 70)
from bio_analyzer import BiologicalDataAnalyzer

bio = BiologicalDataAnalyzer()
sequences = bio.generate_sample_sequences(n_sequences=5)

print(f"Generated {len(sequences)} DNA sequences")
for i, seq in enumerate(sequences[:2]):
    analysis = bio.analyze_sequence(seq)
    print(f"\nSequence {i+1}:")
    print(f"  Length: {analysis['length']} bp")
    print(f"  GC Content: {analysis['gc_content']:.2f}%")
    print(f"  DNA: {str(seq)[:40]}...")
    print(f"  RNA: {analysis['rna_sequence'][:40]}...")

# Example 4: Make Predictions with Trained Model
print("\n\nExample 4: Disease Prediction")
print("-" * 70)
from dl_model import DiseasePredictionModel
import pandas as pd

# Check if model exists
if os.path.exists('models/disease_prediction_model.h5'):
    model = DiseasePredictionModel()
    model.load_model()
    
    # Create sample patient
    sample_patient = pd.DataFrame({
        'age': [65],
        'gender': [1],
        'blood_pressure_systolic': [160],
        'blood_pressure_diastolic': [95],
        'heart_rate': [85],
        'body_temperature': [98.6],
        'cholesterol': [240],
        'glucose': [140],
        'bmi': [32],
        'smoking': [1],
        'exercise_hours': [1],
        'alcohol_consumption': [2],
        'family_history': [1],
        'previous_heart_issues': [1],
        'diabetes_history': [1]
    })
    
    prediction = model.predict(sample_patient)[0]
    probability = model.predict_proba(sample_patient)[0]
    
    print("Patient Profile:")
    print(f"  Age: 65, BP: 160/95, Cholesterol: 240, Glucose: 140")
    print(f"  Smoker: Yes, BMI: 32, Family History: Yes")
    print(f"\nPrediction: {'Disease Risk Detected' if prediction == 1 else 'Low Risk'}")
    print(f"Probability: {probability:.2%}")
else:
    print("Model not found. Run 'python main.py' first to train the model.")

print("\n" + "="*70)
print("For full analysis, run: python main.py")
print("="*70)
