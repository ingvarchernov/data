#!/usr/bin/env python3
"""
ML Price Predictor - Simple machine learning model for price direction prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MLPricePredictor:
    """Machine learning model for predicting price direction based on technical indicators"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'ema_9', 'ema_21', 'ema_50',
            'bb_upper', 'bb_lower', 'bb_middle',
            'volume_ratio', 'price_change_1h', 'price_change_4h',
            'volatility_1h', 'volatility_4h'
        ]
        self.target_column = 'future_direction'  # 1 for up, 0 for down

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features for ML model"""
        try:
            # Calculate additional features
            df = df.copy()

            # Price changes
            df['price_change_1h'] = df['close'].pct_change(4)  # Assuming 15m data
            df['price_change_4h'] = df['close'].pct_change(16)

            # Volatility (standard deviation of returns)
            df['volatility_1h'] = df['close'].pct_change().rolling(4).std()
            df['volatility_4h'] = df['close'].pct_change().rolling(16).std()

            # Volume ratio (current vs average)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Future direction (target) - price change in next 4 hours
            df['future_direction'] = (df['close'].shift(-16) > df['close']).astype(int)

            # Drop NaN values
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train the ML model"""
        try:
            # Prepare features
            df_prepared = self.prepare_features(df)
            if df_prepared.empty:
                return {'success': False, 'error': 'No data after preparation'}

            # Split features and target
            X = df_prepared[self.feature_columns]
            y = df_prepared[self.target_column]

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

            return {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
            }

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Make prediction for current market data"""
        try:
            if self.model is None or self.scaler is None:
                return None

            # Prepare features
            df_prepared = self.prepare_features(df)
            if df_prepared.empty or len(df_prepared) < 1:
                return None

            # Get latest data
            latest_features = df_prepared[self.feature_columns].iloc[-1:]

            # Scale and predict
            features_scaled = self.scaler.transform(latest_features)
            prediction_proba = self.model.predict_proba(features_scaled)[0]

            # Prediction: 1 = UP, 0 = DOWN
            prediction = int(prediction_proba[1] > 0.5)
            confidence = max(prediction_proba) * 100

            return {
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': confidence,
                'probabilities': {
                    'UP': prediction_proba[1] * 100,
                    'DOWN': prediction_proba[0] * 100
                }
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def save_model(self, filepath: str):
        """Save trained model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False