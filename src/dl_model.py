"""
Deep Learning Model for Disease Prediction
Neural network with target accuracy of 92%+
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
import os


class DiseasePredictionModel:
    """Deep Learning model for disease prediction"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build deep neural network architecture optimized for 92%+ accuracy
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer with dropout
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Second hidden layer
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Third hidden layer
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Fourth hidden layer
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Fifth hidden layer
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with optimized learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, epochs: int = 100,
              batch_size: int = 32) -> dict:
        """
        Train the deep learning model
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.000001
        )
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate on test set
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.flatten()
        }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save_model(self, model_path: str = 'models/disease_prediction_model.h5',
                   scaler_path: str = 'models/scaler.pkl'):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, model_path: str = 'models/disease_prediction_model.h5',
                   scaler_path: str = 'models/scaler.pkl'):
        """Load trained model and scaler"""
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
