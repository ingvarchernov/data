#!/usr/bin/env python3
"""
ML Price Predictor - LSTM-based machine learning model for price direction prediction
Updated to use trained LSTM model instead of RandomForest
Adapts to work with the new LSTM training system from train_ml_models.py
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging
import pickle
import json

from config import LSTM_MODEL_CONFIG

logger = logging.getLogger(__name__)


class LSTMPricePredictor:
    """LSTM-based ML model for predicting price direction based on technical indicators"""

    def __init__(self, model_path: Optional[str] = None):
        resolved_model_path = model_path or LSTM_MODEL_CONFIG.get('model_path', 'models/lstm_model.pt')
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = LSTM_MODEL_CONFIG.get('sequence_length', 64)
        self.feature_columns = LSTM_MODEL_CONFIG.get('feature_columns', [])
        self.target_column = 'future_direction'  # 1 for up, 0 for down
        self.model_metadata = None
        
        # Attempt to load model on init
        if Path(resolved_model_path).exists():
            self.load_lstm_model(resolved_model_path)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features for ML model - matches training set exactly"""
        try:
            import ta
            from pattern_detector import calculate_indicators, detect_breakouts
            
            df = df.copy().sort_index()
            
            # Ensure required columns
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Missing column: {col}")
                    return pd.DataFrame()
            
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float).replace(0, 1.0)
            
            feat = pd.DataFrame(index=df.index)
            
            # Price/momentum features (returns and spreads)
            feat["ret_1"] = close.pct_change(1)
            feat["ret_4"] = close.pct_change(4)
            feat["ret_12"] = close.pct_change(12)
            feat["ret_24"] = close.pct_change(24)
            feat["log_ret_1"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0)
            feat["hl_spread"] = (high - low) / close
            feat["oc_spread"] = (df["open"].astype(float) - close) / close
            feat["volatility_12"] = feat["ret_1"].rolling(12).std()
            feat["volatility_48"] = feat["ret_1"].rolling(48).std()
            
            # Volume/flow features
            feat["volume_z_20"] = ((volume - volume.rolling(20).mean()) / volume.rolling(20).std()).fillna(0)
            feat["volume_ratio_20"] = volume / volume.rolling(20).mean()
            feat["trades_ratio_20"] = 1.0  # Fallback if no trades column
            
            # Technical indicators using ta library
            try:
                feat["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            except:
                feat["rsi"] = 50.0
            
            try:
                macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
                feat["macd"] = macd_indicator.macd()
                feat["macd_signal"] = macd_indicator.macd_signal()
                feat["macd_hist"] = macd_indicator.macd_diff()
            except:
                feat["macd"] = 0.0
                feat["macd_signal"] = 0.0
                feat["macd_hist"] = 0.0
            
            try:
                bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
                feat["bb_upper"] = bb.bollinger_hband()
                feat["bb_lower"] = bb.bollinger_lband()
                feat["bb_sma"] = bb.bollinger_mavg()
            except:
                feat["bb_upper"] = close
                feat["bb_lower"] = close
                feat["bb_sma"] = close
            
            try:
                feat["ema20"] = close.ewm(span=20, adjust=False).mean()
                feat["ema50"] = close.ewm(span=50, adjust=False).mean()
            except:
                feat["ema20"] = close
                feat["ema50"] = close
            
            # Price vs MA features
            feat["price_vs_ema20"] = (close - feat["ema20"]) / close.replace(0, 1)
            feat["price_vs_ema50"] = (close - feat["ema50"]) / close.replace(0, 1)
            feat["ema_gap"] = (feat["ema20"] - feat["ema50"]) / close.replace(0, 1)
            
            # Bollinger Band width
            feat["bb_width"] = ((feat["bb_upper"] - feat["bb_lower"]) / feat["bb_sma"].replace(0, 1)).fillna(0.1)
            
            # ATR
            try:
                feat["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
            except:
                feat["atr_14"] = 0.01
            
            # Normalize BB and MACD by close price
            for col in ("ema20", "ema50", "bb_upper", "bb_lower", "bb_sma"):
                if col in feat.columns:
                    feat[col] = feat[col] / close.replace(0, 1)
            
            for col in ("macd", "macd_signal", "macd_hist"):
                if col in feat.columns:
                    feat[col] = feat[col] / close.replace(0, 1)
            
            # Breakout signals (fallback to 0 if not available)
            feat["breakout_confidence"] = 0.0
            feat["breakout_short_signal"] = 0.0
            
            # Replace inf and NaN
            feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
            
            return feat
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def load_lstm_model(self, model_path: Optional[str] = None) -> bool:
        """Load pre-trained LSTM model from checkpoint"""
        try:
            resolved_model_path = model_path or LSTM_MODEL_CONFIG.get('model_path', 'models/lstm_model.pt')

            if not Path(resolved_model_path).exists():
                logger.error(f"LSTM model not found at {resolved_model_path}")
                return False

            checkpoint = torch.load(resolved_model_path, map_location=self.device, weights_only=False)
            
            # Import model class from train_ml_models
            from train_ml_models import LSTMRegressor
            
            # Get config from checkpoint
            config_dict = checkpoint.get('config', {})
            
            input_size = config_dict.get('input_size', len(self.feature_columns))
            
            # Better approach: get actual input size from model weights in checkpoint
            lstm_weights = checkpoint.get('model_state_dict', {})
            for key in lstm_weights:
                if 'lstm.weight_ih' in key:
                    # weight_ih shape is [4*hidden_size, input_size]
                    actual_input_size = lstm_weights[key].shape[1]
                    input_size = actual_input_size
                    logger.info(f"Detected input_size from checkpoint: {input_size}")
                    break
            
            # Reconstruct model with same config as training
            self.model = LSTMRegressor(
                input_size=input_size,
                hidden_size=config_dict.get('hidden_size', 128),
                num_layers=config_dict.get('num_layers', 3),
                dropout=config_dict.get('dropout', 0.25),
                dense_size=config_dict.get('dense_size', 96)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load scaler
            scaler_path = Path(resolved_model_path).parent / "lstm_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler loaded")
            else:
                logger.warning("⚠️ Scaler not found, predictions may be inaccurate")
                
            # Load metadata
            metadata_path = Path(resolved_model_path).parent / "lstm_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.feature_columns = self.model_metadata.get('feature_names', self.feature_columns)
                    self.sequence_length = self.model_metadata.get('sequence_length', 64)
                    logger.info(f"✅ LSTM metadata loaded: {self.model_metadata.get('trained_at')}")
            
            logger.info(f"✅ LSTM model loaded from {resolved_model_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading LSTM model: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.model is not None and self.scaler is not None

    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Make prediction using LSTM model for current market data"""
        try:
            if not self.is_ready():
                logger.error("❌ LSTM model not loaded. Call load_lstm_model() first.")
                return None

            if len(df) < self.sequence_length:
                logger.warning(f"⚠️ Not enough data: need {self.sequence_length}, have {len(df)}")
                return None

            # Prepare features
            df_prepared = self.prepare_features(df)
            if df_prepared.empty or len(df_prepared) < self.sequence_length:
                return None

            # Get last sequence_length rows with available features
            try:
                sequence_data = df_prepared[self.feature_columns].iloc[-self.sequence_length:].values
            except KeyError as e:
                logger.error(f"Missing feature column: {e}")
                return None
            
            # Scale if scaler available
            if self.scaler:
                sequence_data = self.scaler.transform(sequence_data)
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(sequence_tensor)
                output_np = output.cpu().numpy()
                # Handle both 1D [1] and 2D [1, 1] output shapes from squeeze
                prediction_value = float(output_np.flatten()[0])
            
            # Convert output to UP/DOWN
            # LSTM outputs regression value (return), so we threshold at 0
            prediction = 'UP' if prediction_value > 0 else 'DOWN'
            confidence = min(abs(prediction_value) * 100, 100)  # Clamp to 100
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'raw_output': float(prediction_value),
                'probabilities': {
                    'UP': max(0, min(100, (1 + prediction_value) * 50)),
                    'DOWN': max(0, min(100, (1 - prediction_value) * 50))
                }
            }

        except Exception as e:
            logger.error(f"❌ Error making LSTM prediction: {e}")
            return None

    def predict_sequence(self, df: pd.DataFrame, steps: int = 24) -> Optional[List[float]]:
        """
        Make multi-step predictions using LSTM model
        
        Args:
            df: DataFrame with OHLCV data
            steps: Number of future steps to predict
            
        Returns:
            List of predicted returns or None
        """
        try:
            if not self.is_ready():
                logger.error("❌ LSTM model not loaded")
                return None

            if len(df) < self.sequence_length:
                logger.warning(f"⚠️ Not enough data for sequence prediction")
                return None

            predictions = []
            
            # Prepare initial sequence
            df_prepared = self.prepare_features(df)
            
            try:
                current_sequence = df_prepared[self.feature_columns].iloc[-self.sequence_length:].values
            except KeyError:
                return None
            
            for _ in range(steps):
                # Scale
                if self.scaler:
                    seq_scaled = self.scaler.transform(current_sequence)
                else:
                    seq_scaled = current_sequence
                
                # Predict
                seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                
                predictions.append(float(pred))
            
            return predictions

        except Exception as e:
            logger.error(f"❌ Error making sequence prediction: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self.is_ready():
            return {'status': 'not_loaded'}
        
        return {
            'status': 'ready',
            'device': self.device,
            'sequence_length': self.sequence_length,
            'feature_count': len(self.feature_columns),
            'metadata': self.model_metadata
        }


# Backward compatibility alias
MLPricePredictor = LSTMPricePredictor
