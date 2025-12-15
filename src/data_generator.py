"""
HealthInsight AI: Medical Data Analyzer
Synthetic medical dataset generator for disease prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple


class MedicalDataGenerator:
    """Generate synthetic medical data for disease prediction"""
    
    def __init__(self, n_samples: int = 15000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_patient_data(self) -> pd.DataFrame:
        """
        Generate synthetic patient data with medical features
        
        Returns:
            DataFrame with patient features and disease labels
        """
        # Patient demographics
        age = np.random.normal(50, 15, self.n_samples).clip(18, 95)
        gender = np.random.choice([0, 1], self.n_samples)  # 0: Female, 1: Male
        
        # Vital signs
        blood_pressure_systolic = np.random.normal(130, 20, self.n_samples).clip(90, 200)
        blood_pressure_diastolic = np.random.normal(85, 15, self.n_samples).clip(60, 120)
        heart_rate = np.random.normal(75, 12, self.n_samples).clip(50, 120)
        body_temperature = np.random.normal(98.6, 1.2, self.n_samples).clip(96, 104)
        
        # Lab results
        cholesterol = np.random.normal(200, 40, self.n_samples).clip(120, 350)
        glucose = np.random.normal(100, 25, self.n_samples).clip(60, 250)
        bmi = np.random.normal(26, 5, self.n_samples).clip(15, 45)
        
        # Lifestyle factors
        smoking = np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3])
        exercise_hours = np.random.exponential(2, self.n_samples).clip(0, 15)
        alcohol_consumption = np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.3, 0.4, 0.2, 0.1])
        
        # Medical history
        family_history = np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4])
        previous_heart_issues = np.random.choice([0, 1], self.n_samples, p=[0.8, 0.2])
        diabetes_history = np.random.choice([0, 1], self.n_samples, p=[0.75, 0.25])
        
        # Disease prediction (target variable)
        # Enhanced logic with stronger signals for 92%+ accuracy
        risk_score = (
            0.025 * age +
            0.10 * blood_pressure_systolic +
            0.18 * (cholesterol - 200) / 40 +
            0.25 * (glucose - 100) / 25 +
            0.18 * (bmi - 26) / 5 +
            3.5 * smoking +
            -0.35 * exercise_hours +
            1.0 * alcohol_consumption +
            2.5 * family_history +
            4.0 * previous_heart_issues +
            3.0 * diabetes_history +
            np.random.normal(0, 0.8, self.n_samples)
        )
        
        # Convert risk score to binary disease label
        disease = (risk_score > np.percentile(risk_score, 60)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'gender': gender,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rate,
            'body_temperature': body_temperature,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'bmi': bmi,
            'smoking': smoking,
            'exercise_hours': exercise_hours,
            'alcohol_consumption': alcohol_consumption,
            'family_history': family_history,
            'previous_heart_issues': previous_heart_issues,
            'diabetes_history': diabetes_history,
            'disease': disease
        })
        
        return df
    
    def generate_medical_notes(self, n_notes: int = 100) -> list:
        """
        Generate synthetic medical notes for NLP analysis
        
        Returns:
            List of medical text notes
        """
        symptoms = [
            "chest pain", "shortness of breath", "fatigue", "dizziness",
            "headache", "nausea", "fever", "cough", "abdominal pain"
        ]
        
        conditions = [
            "hypertension", "diabetes", "cardiovascular disease", "asthma",
            "pneumonia", "influenza", "gastritis", "migraine"
        ]
        
        notes = []
        for _ in range(n_notes):
            symptom = np.random.choice(symptoms, np.random.randint(1, 4), replace=False)
            condition = np.random.choice(conditions)
            
            note = f"Patient presents with {', '.join(symptom)}. "
            note += f"Preliminary diagnosis suggests {condition}. "
            note += "Recommended further tests and monitoring."
            notes.append(note)
        
        return notes
