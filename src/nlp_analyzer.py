"""
NLP Module for Medical Text Analysis
Processes medical notes and extracts insights
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import numpy as np


class MedicalNLPAnalyzer:
    """NLP analysis for medical text data"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()
        
        # Medical keywords
        self.symptom_keywords = [
            'pain', 'fever', 'cough', 'fatigue', 'nausea', 'dizziness',
            'headache', 'shortness', 'breath', 'chest', 'abdominal'
        ]
        
        self.condition_keywords = [
            'hypertension', 'diabetes', 'cardiovascular', 'asthma',
            'pneumonia', 'influenza', 'disease', 'syndrome'
        ]
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    def preprocess_text(self, text: str) -> list:
        """
        Preprocess medical text
        
        Args:
            text: Raw medical text
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def extract_symptoms(self, text: str) -> list:
        """
        Extract symptoms from medical text
        
        Args:
            text: Medical text
            
        Returns:
            List of identified symptoms
        """
        tokens = self.preprocess_text(text)
        symptoms = [t for t in tokens if t in self.symptom_keywords]
        return list(set(symptoms))
    
    def extract_conditions(self, text: str) -> list:
        """
        Extract medical conditions from text
        
        Args:
            text: Medical text
            
        Returns:
            List of identified conditions
        """
        tokens = self.preprocess_text(text)
        conditions = [t for t in tokens if t in self.condition_keywords]
        return list(set(conditions))
    
    def analyze_notes(self, notes: list) -> dict:
        """
        Analyze a collection of medical notes
        
        Args:
            notes: List of medical text notes
            
        Returns:
            Dictionary with analysis results
        """
        all_symptoms = []
        all_conditions = []
        all_tokens = []
        
        for note in notes:
            symptoms = self.extract_symptoms(note)
            conditions = self.extract_conditions(note)
            tokens = self.preprocess_text(note)
            
            all_symptoms.extend(symptoms)
            all_conditions.extend(conditions)
            all_tokens.extend(tokens)
        
        # Count occurrences
        symptom_counts = Counter(all_symptoms)
        condition_counts = Counter(all_conditions)
        word_counts = Counter(all_tokens)
        
        results = {
            'total_notes': len(notes),
            'unique_symptoms': len(symptom_counts),
            'unique_conditions': len(condition_counts),
            'most_common_symptoms': symptom_counts.most_common(10),
            'most_common_conditions': condition_counts.most_common(10),
            'most_common_words': word_counts.most_common(20),
            'avg_note_length': np.mean([len(note.split()) for note in notes])
        }
        
        return results
    
    def sentiment_analysis(self, text: str) -> str:
        """
        Simple sentiment analysis for medical notes
        
        Args:
            text: Medical text
            
        Returns:
            Sentiment category
        """
        # Simple keyword-based sentiment
        negative_words = ['severe', 'critical', 'emergency', 'urgent', 'deteriorating']
        positive_words = ['improving', 'stable', 'recovery', 'healthy', 'normal']
        
        text_lower = text.lower()
        
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if neg_count > pos_count:
            return 'concerning'
        elif pos_count > neg_count:
            return 'positive'
        else:
            return 'neutral'
