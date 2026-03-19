#!/usr/bin/env python3
"""
Ensemble Predictor - LSTM-based
Combines LSTM predictions with pattern analysis and market regime for robust forecasting
Synchronized with new LSTM training system from train_ml_models.py
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from config import ENSEMBLE_CONFIG

logger = logging.getLogger(__name__)


class LSTMEnsemblePredictor:
    """Combines LSTM predictions, pattern analysis, and market regime"""

    def __init__(self, lstm_model_path: Optional[str] = None):
        model_path = lstm_model_path or ENSEMBLE_CONFIG.get('ml_model_path', 'models/lstm_model.pt')

        try:
            from ml_predictor_lstm import LSTMPricePredictor
            self.ml_predictor = LSTMPricePredictor(model_path)
            logger.info("✅ Loaded LSTMPricePredictor from ml_predictor_lstm")
        except Exception as e:
            logger.warning(f"⚠️ Could not load LSTM predictor: {e}")
            self.ml_predictor = None

        self.weights = ENSEMBLE_CONFIG.get('weights', {
            'pattern_confidence': 0.35,
            'ml_prediction': 0.40,
            'market_regime': 0.15,
            'technical_signals': 0.10
        })
        self.decision_threshold = ENSEMBLE_CONFIG.get('decision_threshold', 0.55)

    def load_lstm_model(self, model_path: str) -> bool:
        """Load LSTM model"""
        try:
            if self.ml_predictor and hasattr(self.ml_predictor, 'load_lstm_model'):
                return self.ml_predictor.load_lstm_model(model_path)
            return False
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            return False

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for ensemble"""
        try:
            if len(df) < 50:
                return {'score': 0.5, 'signals': []}

            signals = []
            score = 0.5  # Neutral baseline

            # Try using Rust indicators for speed
            try:
                from pattern_detector import calculate_indicators
                indicators = calculate_indicators(df['close'].values.tolist())
                
                # Get latest values
                rsi = indicators.get('rsi', [50.0])[-1] if indicators.get('rsi') else 50.0
                macd_hist = indicators.get('macd_histogram', [0.0])[-1] if indicators.get('macd_histogram') else 0.0
                macd_hist_prev = indicators.get('macd_histogram', [0.0])[-2] if len(indicators.get('macd_histogram', [])) > 1 else 0.0
                
            except (ImportError, Exception):
                # Fallback to pandas_ta
                logger.debug("Using pandas_ta for technical indicators")
                import pandas_ta as ta
                
                rsi_vals = ta.rsi(df['close'], length=14)
                rsi = rsi_vals.iloc[-1] if len(rsi_vals) > 0 else 50.0
                
                macd_result = ta.macd(df['close'])
                macd_hist = macd_result.iloc[-1, 2] if not macd_result.empty else 0.0
                macd_hist_prev = macd_result.iloc[-2, 2] if len(macd_result) > 1 else 0.0

            # RSI signals
            if rsi < 30:
                signals.append('rsi_oversold')
                score += 0.1  # Bullish
            elif rsi > 70:
                signals.append('rsi_overbought')
                score -= 0.1  # Bearish

            # MACD signals
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                signals.append('macd_bullish')
                score += 0.05
            elif macd_hist < 0 and macd_hist < macd_hist_prev:
                signals.append('macd_bearish')
                score -= 0.05

            # Volume confirmation (if available)
            if 'volume' in df.columns and len(df) > 1:
                volume_ratio = df['volume'].iloc[-1] / df['volume'].iloc[:-1].mean()
                if volume_ratio > 1.5:
                    signals.append('high_volume')
                    score += 0.05 if score > 0.5 else -0.05

            return {
                'score': np.clip(score, 0, 1),
                'signals': signals
            }

        except Exception as e:
            logger.error(f"Error calculating technical signals: {e}")
            return {'score': 0.5, 'signals': []}

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        pattern_confidence: float,
        pattern_direction: str
    ) -> Dict:
        """
        Generate ensemble prediction combining LSTM, patterns, and market regime
        
        Args:
            df: Market data DataFrame with OHLCV
            pattern_confidence: Pattern confidence (0-100)
            pattern_direction: 'LONG' or 'SHORT'
            
        Returns:
            Ensemble prediction with confidence and components breakdown
        """
        try:
            components = {}

            # 1. Pattern confidence (normalized to 0-1)
            pattern_score = pattern_confidence / 100.0
            if pattern_direction == 'SHORT':
                pattern_score = 1 - pattern_score  # Invert for bearish
            components['pattern'] = pattern_score

            # 2. LSTM ML prediction (35-40% weight)
            ml_score = 0.5  # Neutral default
            if self.ml_predictor:
                try:
                    # Check if model is ready (for new LSTMPricePredictor)
                    if hasattr(self.ml_predictor, 'is_ready'):
                        if not self.ml_predictor.is_ready():
                            logger.debug("⚠️ LSTM model not ready, using neutral prediction")
                    
                    ml_pred = self.ml_predictor.predict(df)
                    if ml_pred:
                        # Use the probability from LSTM prediction
                        ml_score = ml_pred.get('probabilities', {}).get('UP', 50) / 100.0
                        logger.debug(f"LSTM prediction: {ml_pred['prediction']} (confidence: {ml_pred['confidence']:.1f}%)")
                    else:
                        logger.debug("LSTM returned no prediction")
                except Exception as e:
                    logger.debug(f"LSTM prediction error: {e}, using neutral")
            
            components['ml'] = ml_score

            # 3. Market regime analysis
            regime_score = 0.5  # Neutral default
            try:
                from pattern_analytics import PatternAnalytics, MarketRegime
                analytics = PatternAnalytics()
                regime = analytics.detect_market_regime(df)
                
                if regime == MarketRegime.TRENDING_UP:
                    regime_score = 0.7
                elif regime == MarketRegime.TRENDING_DOWN:
                    regime_score = 0.3
                else:  # RANGING or VOLATILE
                    regime_score = 0.5
                    
                components['regime'] = regime_score
                components['regime_str'] = regime.value
            except Exception as e:
                logger.debug(f"Could not analyze market regime: {e}")
                components['regime'] = regime_score
                components['regime_str'] = 'UNKNOWN'

            # 4. Technical signals
            tech_signals = self.calculate_technical_signals(df)
            components['technical'] = tech_signals['score']

            # Calculate weighted ensemble score
            ensemble_score = (
                components['pattern'] * self.weights['pattern_confidence'] +
                components['ml'] * self.weights['ml_prediction'] +
                components['regime'] * self.weights['market_regime'] +
                components['technical'] * self.weights['technical_signals']
            )

            # Normalize to 0-1 range
            ensemble_score = np.clip(ensemble_score, 0, 1)

            # Final prediction and confidence
            prediction = 'UP' if ensemble_score > self.decision_threshold else 'DOWN'
            confidence = abs(ensemble_score - 0.5) * 200  # Scale to 0-100

            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'ensemble_score': float(ensemble_score),
                'components': {k: float(v) for k, v in components.items() if isinstance(v, (int, float))},
                'component_weights': self.weights,
                'technical_signals': tech_signals['signals']
            }

        except Exception as e:
            logger.error(f"❌ Ensemble prediction error: {e}")
            return {
                'prediction': pattern_direction,
                'confidence': pattern_confidence,
                'ensemble_score': 0.5,
                'components': {},
                'error': str(e)
            }

    def get_ensemble_info(self) -> Dict:
        """Get information about ensemble configuration"""
        info = {
            'weights': self.weights,
            'decision_threshold': self.decision_threshold,
            'ml_model_ready': False,
            'ml_model_info': None
        }
        
        if self.ml_predictor:
            if hasattr(self.ml_predictor, 'is_ready'):
                info['ml_model_ready'] = self.ml_predictor.is_ready()
                if hasattr(self.ml_predictor, 'get_model_info'):
                    info['ml_model_info'] = self.ml_predictor.get_model_info()
            elif hasattr(self.ml_predictor, 'model'):
                info['ml_model_ready'] = self.ml_predictor.model is not None
        
        return info


# Backward compatibility alias
EnsemblePredictor = LSTMEnsemblePredictor
