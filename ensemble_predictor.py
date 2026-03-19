#!/usr/bin/env python3
"""
Ensemble Predictor - Combines multiple signals for better predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from ml_predictor import MLPricePredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combines pattern analysis, ML predictions, and market regime for ensemble forecasting"""

    def __init__(self):
        self.ml_predictor = MLPricePredictor()
        self.weights = {
            'pattern_confidence': 0.4,
            'ml_prediction': 0.35,
            'market_regime': 0.15,
            'technical_signals': 0.1
        }

    def load_ml_model(self, model_path: str) -> bool:
        """Load pre-trained ML model"""
        return self.ml_predictor.load_model(model_path)

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for ensemble"""
        try:
            if len(df) < 50:
                return {'score': 0.5, 'signals': []}

            recent = df.tail(20)

            signals = []
            score = 0.5  # Neutral

            # Use Rust indicators for speed
            from pattern_detector import calculate_indicators
            indicators = calculate_indicators(df['close'].values.tolist())
            
            # Get latest values
            rsi = indicators['rsi'][-1] if indicators['rsi'] else 50.0
            macd_hist = indicators['macd_histogram'][-1] if indicators['macd_histogram'] else 0.0
            macd_hist_prev = indicators['macd_histogram'][-2] if len(indicators['macd_histogram']) > 1 else 0.0
            ema9 = indicators['ema20'][-1] if indicators['ema20'] else df['close'].iloc[-1]
            ema21 = indicators['ema50'][-1] if indicators['ema50'] else df['close'].iloc[-1]

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

            # EMA alignment
            if ema9 > ema21:
                signals.append('ema_bullish')
                score += 0.05
            else:
                signals.append('ema_bearish')
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
        Generate ensemble prediction combining all signals

        Args:
            df: Market data DataFrame
            pattern_confidence: Pattern confidence (0-100)
            pattern_direction: 'LONG' or 'SHORT'

        Returns:
            Ensemble prediction with confidence and components
        """
        try:
            components = {}

            # 1. Pattern confidence (normalized)
            pattern_score = pattern_confidence / 100.0
            if pattern_direction == 'SHORT':
                pattern_score = 1 - pattern_score  # Invert for bearish
            components['pattern'] = pattern_score

            # 2. ML prediction
            ml_pred = self.ml_predictor.predict(df)
            if ml_pred:
                ml_score = ml_pred['probabilities']['UP'] / 100.0
                components['ml'] = ml_score
            else:
                components['ml'] = 0.5  # Neutral if no ML

            # 3. Market regime (from pattern analytics)
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

            # Final prediction
            prediction = 'UP' if ensemble_score > 0.55 else 'DOWN'
            confidence = abs(ensemble_score - 0.5) * 200  # 0-100 scale

            return {
                'prediction': prediction,
                'confidence': confidence,
                'ensemble_score': ensemble_score,
                'components': components,
                'regime': regime.value,
                'technical_signals': tech_signals['signals']
            }

        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {
                'prediction': pattern_direction,
                'confidence': pattern_confidence,
                'ensemble_score': 0.5,
                'components': {},
                'error': str(e)
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (fallback method)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD (fallback method)"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal

        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else 0
        }