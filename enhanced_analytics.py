#!/usr/bin/env python3
"""
Enhanced Analytics Module
Market regime detection, ensemble predictions, and risk-adjusted position sizing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class MLFeatures:
    """Features for ML prediction"""
    # Price action features
    returns_1h: float
    returns_4h: float
    returns_24h: float
    volatility_1h: float
    volatility_24h: float

    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    atr: float

    # Volume features
    volume_ratio: float
    volume_ma_ratio: float

    # Pattern features
    pattern_confidence: float
    pattern_type_encoded: int
    pattern_direction_encoded: int

    # Market structure
    support_distance: float
    resistance_distance: float
    trend_strength: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.returns_1h, self.returns_4h, self.returns_24h,
            self.volatility_1h, self.volatility_24h,
            self.rsi, self.macd, self.macd_signal,
            self.bb_upper, self.bb_lower, self.bb_middle, self.atr,
            self.volume_ratio, self.volume_ma_ratio,
            self.pattern_confidence, self.pattern_type_encoded, self.pattern_direction_encoded,
            self.support_distance, self.resistance_distance, self.trend_strength
        ])


@dataclass
class PredictionResult:
    """ML prediction result"""
    predicted_return_1h: float
    predicted_return_4h: float
    predicted_return_24h: float
    confidence_score: float
    feature_importance: Dict[str, float]
    model_used: str


@dataclass
class EnsembleSignal:
    """Ensemble prediction combining multiple signals"""
    direction: str
    confidence: float
    ml_prediction: float
    pattern_score: float
    regime_score: float
    risk_adjustment: float
    final_score: float


class MarketRegimeDetector:
    """Detect market regime (trending vs ranging)"""

    def __init__(self):
        self.adx_period = 14
        self.atr_period = 14
        self.ranging_threshold = 25
        self.trending_threshold = 30

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Calculate ADX for trend strength
            high = df['high']
            low = df['low']
            close = df['close']

            # Simplified ADX calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(self.atr_period).mean()

            # Directional Movement
            dm_plus = np.where((high - high.shift()) > (low.shift() - low),
                             np.maximum(high - high.shift(), 0), 0)
            dm_minus = np.where((low.shift() - low) > (high - high.shift()),
                               np.maximum(low.shift() - low, 0), 0)

            di_plus = 100 * pd.Series(dm_plus).rolling(self.adx_period).mean() / atr
            di_minus = 100 * pd.Series(dm_minus).rolling(self.adx_period).mean() / atr

            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(self.adx_period).mean()

            # Volatility ratio
            volatility = atr / close * 100

            # Get latest values
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
            current_volatility = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 2

            # Determine regime
            if current_adx > self.trending_threshold:
                # Strong trend - determine direction
                recent_returns = (close.iloc[-10:] / close.iloc[-10]).pct_change().sum()
                return MarketRegime.TRENDING_UP if recent_returns > 0 else MarketRegime.TRENDING_DOWN
            elif current_adx < self.ranging_threshold and current_volatility < 3:
                return MarketRegime.RANGING
            else:
                return MarketRegime.VOLATILE

        except Exception as e:
            logger.warning(f"Regime detection error: {e}")
            return MarketRegime.RANGING








class EnsemblePredictor:
    """Ensemble prediction combining multiple signals"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()

    def generate_ensemble_signal(self, df: pd.DataFrame, pattern_data: Dict) -> EnsembleSignal:
        """Generate ensemble signal combining all analysis methods"""
        try:
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(df)

            # 2. Calculate component scores
            pattern_score = pattern_data.get('confidence', 50)
            regime_score = self._calculate_regime_score(regime, pattern_data)

            # 3. Risk adjustment based on volatility and regime
            risk_adjustment = self._calculate_risk_adjustment(df, regime)

            # 4. Ensemble final score (weighted combination)
            weights = {
                'pattern': 0.5,
                'regime': 0.3,
                'risk': 0.2
            }

            final_score = (
                pattern_score * weights['pattern'] +
                regime_score * weights['regime'] +
                risk_adjustment * weights['risk']
            )

            direction = pattern_data.get('direction', 'LONG')

            return EnsembleSignal(
                direction=direction,
                confidence=pattern_score,
                ml_prediction=0,
                pattern_score=pattern_score,
                regime_score=regime_score,
                risk_adjustment=risk_adjustment,
                final_score=final_score
            )

        except Exception as e:
            logger.warning(f"Ensemble prediction error: {e}")
            return EnsembleSignal(
                direction=pattern_data.get('direction', 'LONG'),
                confidence=pattern_data.get('confidence', 50),
                ml_prediction=0,
                pattern_score=50,
                regime_score=50,
                risk_adjustment=50,
                final_score=50
            )

    def _calculate_regime_score(self, regime: MarketRegime, pattern_data: Dict) -> float:
        """Calculate regime suitability score"""
        direction = pattern_data.get('direction', 'LONG')

        if regime == MarketRegime.TRENDING_UP and direction == 'LONG':
            return 80
        elif regime == MarketRegime.TRENDING_DOWN and direction == 'SHORT':
            return 80
        elif regime == MarketRegime.RANGING:
            return 70
        elif regime == MarketRegime.VOLATILE and 'Breakout' in pattern_data.get('pattern_type', ''):
            return 85
        else:
            return 50

    def _calculate_risk_adjustment(self, df: pd.DataFrame, regime: MarketRegime) -> float:
        """Calculate risk adjustment factor"""
        try:
            # Base risk score
            base_risk = 50

            # Volatility adjustment
            returns = df['close'].pct_change()
            volatility = returns.std() * 100

            if volatility > 5:
                base_risk -= 20  # High volatility = higher risk
            elif volatility < 1:
                base_risk += 10  # Low volatility = lower risk

            # Regime adjustment
            if regime == MarketRegime.VOLATILE:
                base_risk -= 15
            elif regime == MarketRegime.RANGING:
                base_risk += 5

            return np.clip(base_risk, 0, 100)

        except:
            return 50


class RiskAdjustedPositionSizer:
    """Risk-adjusted position sizing"""

    def __init__(self, max_risk_per_trade: float = 0.02, max_total_risk: float = 0.06):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_total_risk = max_total_risk  # 6% total exposure

    def calculate_position_size(self, account_balance: float, entry_price: float,
                              stop_loss: float, ensemble_signal: EnsembleSignal,
                              current_exposure: float = 0) -> float:
        """Calculate risk-adjusted position size"""
        try:
            # Risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                return 0

            # Maximum position size based on single trade risk
            max_position_risk = account_balance * self.max_risk_per_trade
            position_size_risk = max_position_risk / risk_per_unit

            # Adjust for ensemble confidence
            confidence_multiplier = ensemble_signal.final_score / 100.0
            confidence_multiplier = np.clip(confidence_multiplier, 0.3, 1.5)  # Min 30%, max 150%

            # Adjust for ML prediction magnitude
            ml_magnitude = abs(ensemble_signal.ml_prediction)
            ml_multiplier = np.clip(ml_magnitude * 10, 0.5, 2.0)  # Scale ML prediction

            # Adjust for volatility (from ML features if available)
            volatility_adjustment = 1.0
            if hasattr(ensemble_signal, 'ml_prediction') and ensemble_signal.ml_prediction != 0:
                # Higher volatility = smaller position
                volatility_adjustment = 1.0 / (1.0 + ml_magnitude * 5)

            # Total exposure check
            available_risk = self.max_total_risk - (current_exposure / account_balance)
            available_risk = max(available_risk, 0)
            exposure_multiplier = available_risk / self.max_risk_per_trade
            exposure_multiplier = np.clip(exposure_multiplier, 0.1, 1.0)

            # Combine all factors
            final_multiplier = (
                confidence_multiplier *
                ml_multiplier *
                volatility_adjustment *
                exposure_multiplier
            )

            position_size = position_size_risk * final_multiplier

            # Final safety checks
            max_position_pct = account_balance * 0.05  # Max 5% of account per position
            max_position_value = max_position_pct / entry_price

            position_size = min(position_size, max_position_value)

            # Minimum position size (0.001 BTC or equivalent)
            min_position = 0.001 * account_balance / entry_price
            position_size = max(position_size, min_position)

            return position_size

        except Exception as e:
            logger.warning(f"Position sizing error: {e}")
            # Fallback to basic calculation
            risk_amount = account_balance * self.max_risk_per_trade
            return risk_amount / abs(entry_price - stop_loss) if stop_loss != entry_price else 0


class EnhancedAnalytics:
    """Main enhanced analytics class combining all features"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.ensemble_predictor = EnsemblePredictor()
        self.position_sizer = RiskAdjustedPositionSizer()

    def analyze_pattern(self, df: pd.DataFrame, pattern_data: Dict) -> Dict[str, Any]:
        """Complete analysis of a pattern with all enhanced features"""
        try:
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(df)

            # 2. Generate ensemble signal
            ensemble_signal = self.ensemble_predictor.generate_ensemble_signal(df, pattern_data)

            # 3. Basic confidence from regime and pattern
            enhanced_confidence = pattern_data.get('confidence', 50)
            if regime == MarketRegime.TRENDING_UP and pattern_data.get('direction') == 'LONG':
                enhanced_confidence += 10
            elif regime == MarketRegime.TRENDING_DOWN and pattern_data.get('direction') == 'SHORT':
                enhanced_confidence += 10

            return {
                'regime': regime.value,
                'ensemble_signal': ensemble_signal,
                'enhanced_confidence': min(enhanced_confidence, 100),
                'ml_prediction': None,
                'features': None,
                'recommendation': self._generate_recommendation(ensemble_signal, regime)
            }

        except Exception as e:
            logger.warning(f"Enhanced analysis error: {e}")
            return {
                'regime': 'ranging',
                'ensemble_signal': None,
                'enhanced_confidence': pattern_data.get('confidence', 50),
                'ml_prediction': None,
                'features': None,
                'recommendation': 'HOLD'
            }

    def _generate_recommendation(self, ensemble_signal: EnsembleSignal,
                               regime: MarketRegime) -> str:
        """Generate trading recommendation"""
        if ensemble_signal and ensemble_signal.final_score > 70:
            return f"STRONG_{ensemble_signal.direction}"
        elif ensemble_signal and ensemble_signal.final_score > 55:
            return f"MODERATE_{ensemble_signal.direction}"
        elif ensemble_signal and ensemble_signal.final_score < 30:
            return "AVOID"
        else:
            return "WEAK_SIGNAL"

    def get_risk_adjusted_position_size(self, account_balance: float, entry_price: float,
                                      stop_loss: float, analysis_result: Dict,
                                      current_exposure: float = 0) -> float:
        """Get risk-adjusted position size"""
        ensemble_signal = analysis_result.get('ensemble_signal')
        if ensemble_signal:
            return self.position_sizer.calculate_position_size(
                account_balance, entry_price, stop_loss, ensemble_signal, current_exposure
            )
        else:
            # Fallback calculation
            risk_amount = account_balance * 0.02
            risk_per_unit = abs(entry_price - stop_loss)
            return risk_amount / risk_per_unit if risk_per_unit > 0 else 0


# Global instance for easy access
enhanced_analytics = EnhancedAnalytics()