#!/usr/bin/env python3
"""
Enhanced Analytics Module
ML-based price prediction, improved confidence scoring, market regime detection,
ensemble predictions, and risk-adjusted position sizing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pickle
import os
from pathlib import Path

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    import ta
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Enhanced analytics will use basic methods.")

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


class MLPredictor:
    """ML-based price prediction using pattern features"""

    def __init__(self, model_path: str = "models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False

        if ML_AVAILABLE:
            self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            # Try to load existing models
            model_files = ["rf_model.pkl", "xgb_model.pkl", "lgb_model.pkl", "scaler.pkl"]
            if all((self.model_path / f).exists() for f in model_files):
                with open(self.model_path / "rf_model.pkl", "rb") as f:
                    self.models["rf"] = pickle.load(f)
                with open(self.model_path / "xgb_model.pkl", "rb") as f:
                    self.models["xgb"] = pickle.load(f)
                with open(self.model_path / "lgb_model.pkl", "rb") as f:
                    self.models["lgb"] = pickle.load(f)
                with open(self.model_path / "scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded existing ML models")
            else:
                # Create new models
                self.models["rf"] = RandomForestRegressor(n_estimators=100, random_state=42)
                self.models["xgb"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
                self.models["lgb"] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
                logger.info("Created new ML models")
        except Exception as e:
            logger.warning(f"Model loading error: {e}")
            self.is_trained = False

    def extract_features(self, df: pd.DataFrame, pattern_confidence: float = 0,
                        pattern_type: str = "", pattern_direction: str = "") -> MLFeatures:
        """Extract ML features from market data"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Price returns
            returns_1h = close.pct_change(1).iloc[-1] if len(close) > 1 else 0
            returns_4h = close.pct_change(4).iloc[-1] if len(close) > 4 else 0
            returns_24h = close.pct_change(24).iloc[-1] if len(close) > 24 else 0

            # Volatility
            volatility_1h = close.pct_change(1).rolling(10).std().iloc[-1] if len(close) > 10 else 0.02
            volatility_24h = close.pct_change(1).rolling(100).std().iloc[-1] if len(close) > 100 else 0.02

            # Technical indicators
            if ML_AVAILABLE:
                rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1] if len(close) > 14 else 50
                macd = ta.trend.MACD(close).macd().iloc[-1] if len(close) > 26 else 0
                macd_signal = ta.trend.MACD(close).macd_signal().iloc[-1] if len(close) > 26 else 0

                bb_indicator = ta.volatility.BollingerBands(close)
                bb_upper = bb_indicator.bollinger_hband().iloc[-1] if len(close) > 20 else close.iloc[-1] * 1.02
                bb_lower = bb_indicator.bollinger_lband().iloc[-1] if len(close) > 20 else close.iloc[-1] * 0.98
                bb_middle = bb_indicator.bollinger_mavg().iloc[-1] if len(close) > 20 else close.iloc[-1]
            else:
                rsi, macd, macd_signal = 50, 0, 0
                bb_upper, bb_lower, bb_middle = close.iloc[-1] * 1.02, close.iloc[-1] * 0.98, close.iloc[-1]

            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if len(tr) > 14 else abs(high.iloc[-1] - low.iloc[-1])

            # Volume features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if len(volume_ma) > 0 and volume_ma.iloc[-1] > 0 else 1
            volume_ma_ratio = volume_ma.iloc[-1] / volume_ma.rolling(50).mean().iloc[-1] if len(volume_ma) > 50 else 1

            # Pattern encoding
            pattern_type_encoded = hash(pattern_type) % 100  # Simple encoding
            pattern_direction_encoded = 1 if pattern_direction == "LONG" else 0

            # Market structure
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()
            current_price = close.iloc[-1]

            support_distance = abs(current_price - recent_low) / current_price
            resistance_distance = abs(current_price - recent_high) / current_price

            # Trend strength (simplified)
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if len(sma_50) > 0 else 0

            return MLFeatures(
                returns_1h=returns_1h,
                returns_4h=returns_4h,
                returns_24h=returns_24h,
                volatility_1h=volatility_1h,
                volatility_24h=volatility_24h,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle,
                atr=atr,
                volume_ratio=volume_ratio,
                volume_ma_ratio=volume_ma_ratio,
                pattern_confidence=pattern_confidence,
                pattern_type_encoded=pattern_type_encoded,
                pattern_direction_encoded=pattern_direction_encoded,
                support_distance=support_distance,
                resistance_distance=resistance_distance,
                trend_strength=trend_strength
            )

        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            # Return default features
            return MLFeatures(
                returns_1h=0, returns_4h=0, returns_24h=0,
                volatility_1h=0.02, volatility_24h=0.02,
                rsi=50, macd=0, macd_signal=0,
                bb_upper=1, bb_lower=1, bb_middle=1, atr=0.01,
                volume_ratio=1, volume_ma_ratio=1,
                pattern_confidence=50, pattern_type_encoded=0, pattern_direction_encoded=0,
                support_distance=0.01, resistance_distance=0.01, trend_strength=0
            )

    def predict(self, features: MLFeatures) -> PredictionResult:
        """Make price prediction using ensemble of ML models"""
        if not self.is_trained or not ML_AVAILABLE:
            # Return basic prediction
            return PredictionResult(
                predicted_return_1h=0.001,
                predicted_return_4h=0.005,
                predicted_return_24h=0.02,
                confidence_score=0.5,
                feature_importance={},
                model_used="basic"
            )

        try:
            # Prepare features
            X = features.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                predictions[name] = pred

            # Ensemble prediction (weighted average)
            weights = {"rf": 0.3, "xgb": 0.4, "lgb": 0.3}
            ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)

            # Calculate confidence based on model agreement
            pred_values = list(predictions.values())
            confidence = 1 - np.std(pred_values) / (abs(np.mean(pred_values)) + 0.001)
            confidence = np.clip(confidence, 0, 1)

            # Feature importance (from Random Forest)
            if hasattr(self.models["rf"], "feature_importances_"):
                feature_names = [
                    "returns_1h", "returns_4h", "returns_24h", "volatility_1h", "volatility_24h",
                    "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_middle", "atr",
                    "volume_ratio", "volume_ma_ratio", "pattern_confidence", "pattern_type_encoded",
                    "pattern_direction_encoded", "support_distance", "resistance_distance", "trend_strength"
                ]
                importance = dict(zip(feature_names, self.models["rf"].feature_importances_))
            else:
                importance = {}

            return PredictionResult(
                predicted_return_1h=ensemble_pred * 0.1,  # Scale down for 1h
                predicted_return_4h=ensemble_pred * 0.3,  # Scale for 4h
                predicted_return_24h=ensemble_pred,       # Full prediction for 24h
                confidence_score=confidence,
                feature_importance=importance,
                model_used="ensemble"
            )

        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return PredictionResult(
                predicted_return_1h=0.001,
                predicted_return_4h=0.005,
                predicted_return_24h=0.02,
                confidence_score=0.3,
                feature_importance={},
                model_used="fallback"
            )


class EnhancedConfidenceScorer:
    """Improved pattern confidence scoring"""

    def __init__(self):
        self.base_weights = {
            'pattern_quality': 0.3,
            'market_conditions': 0.25,
            'volume_confirmation': 0.2,
            'technical_alignment': 0.15,
            'regime_suitability': 0.1
        }

    def calculate_enhanced_confidence(self, df: pd.DataFrame, pattern_data: Dict,
                                    regime: MarketRegime, ml_prediction: Optional[PredictionResult] = None) -> float:
        """Calculate enhanced confidence score"""
        try:
            base_confidence = pattern_data.get('confidence', 50)

            # Pattern quality score (based on pattern characteristics)
            pattern_quality = self._calculate_pattern_quality(pattern_data)

            # Market conditions score
            market_conditions = self._calculate_market_conditions(df)

            # Volume confirmation score
            volume_confirmation = self._calculate_volume_confirmation(df, pattern_data)

            # Technical alignment score
            technical_alignment = self._calculate_technical_alignment(df, pattern_data)

            # Regime suitability score
            regime_suitability = self._calculate_regime_suitability(pattern_data, regime)

            # ML prediction boost (if available)
            ml_boost = 0
            if ml_prediction and ml_prediction.confidence_score > 0.6:
                # Boost confidence if ML prediction aligns with pattern direction
                pattern_direction = pattern_data.get('direction', 'LONG')
                predicted_positive = ml_prediction.predicted_return_24h > 0.01

                if (pattern_direction == 'LONG' and predicted_positive) or \
                   (pattern_direction == 'SHORT' and not predicted_positive):
                    ml_boost = ml_prediction.confidence_score * 10  # Up to 10 points boost

            # Weighted combination
            enhanced_confidence = (
                pattern_quality * self.base_weights['pattern_quality'] +
                market_conditions * self.base_weights['market_conditions'] +
                volume_confirmation * self.base_weights['volume_confirmation'] +
                technical_alignment * self.base_weights['technical_alignment'] +
                regime_suitability * self.base_weights['regime_suitability'] +
                ml_boost * 0.1  # ML boost as additional factor
            )

            # Normalize to 0-100
            enhanced_confidence = np.clip(enhanced_confidence, 0, 100)

            return enhanced_confidence

        except Exception as e:
            logger.warning(f"Enhanced confidence calculation error: {e}")
            return base_confidence

    def _calculate_pattern_quality(self, pattern_data: Dict) -> float:
        """Calculate pattern quality score"""
        pattern_type = pattern_data.get('pattern_type', '')
        base_score = pattern_data.get('confidence', 50)

        # Pattern type multipliers
        type_multipliers = {
            'Compression Breakout': 1.2,
            'Double Top': 0.9,
            'Double Bottom': 0.9,
            'Head & Shoulders': 1.0,
            'Triangle': 1.1,
            'Flag': 1.0
        }

        multiplier = type_multipliers.get(pattern_type, 1.0)
        return min(base_score * multiplier, 100)

    def _calculate_market_conditions(self, df: pd.DataFrame) -> float:
        """Calculate market conditions score"""
        try:
            # Volatility score
            returns = df['close'].pct_change()
            volatility = returns.std() * 100

            # Optimal volatility range: 1-3%
            if 1 <= volatility <= 3:
                vol_score = 80
            elif 0.5 <= volatility <= 5:
                vol_score = 60
            else:
                vol_score = 40

            # Trend strength (using SMA alignment)
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()

            if len(sma_20) > 0 and len(sma_50) > 0:
                trend_alignment = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
                trend_score = min(trend_alignment * 1000, 100)  # Scale and cap
            else:
                trend_score = 50

            return (vol_score + trend_score) / 2

        except:
            return 50

    def _calculate_volume_confirmation(self, df: pd.DataFrame, pattern_data: Dict) -> float:
        """Calculate volume confirmation score"""
        try:
            volume = df['volume']
            avg_volume = volume.tail(20).mean()

            # Check if volume is above average
            current_volume = volume.iloc[-1]
            if current_volume > avg_volume * 1.2:
                return 80
            elif current_volume > avg_volume:
                return 60
            else:
                return 40

        except:
            return 50

    def _calculate_technical_alignment(self, df: pd.DataFrame, pattern_data: Dict) -> float:
        """Calculate technical alignment score"""
        try:
            close = df['close']

            # RSI alignment
            if ML_AVAILABLE:
                rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
                if pattern_data.get('direction') == 'LONG' and rsi < 70:
                    rsi_score = 70
                elif pattern_data.get('direction') == 'SHORT' and rsi > 30:
                    rsi_score = 70
                else:
                    rsi_score = 40
            else:
                rsi_score = 50

            # Moving average alignment
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()

            if len(sma_20) > 0 and len(sma_50) > 0:
                if pattern_data.get('direction') == 'LONG' and sma_20.iloc[-1] > sma_50.iloc[-1]:
                    ma_score = 80
                elif pattern_data.get('direction') == 'SHORT' and sma_20.iloc[-1] < sma_50.iloc[-1]:
                    ma_score = 80
                else:
                    ma_score = 50
            else:
                ma_score = 50

            return (rsi_score + ma_score) / 2

        except:
            return 50

    def _calculate_regime_suitability(self, pattern_data: Dict, regime: MarketRegime) -> float:
        """Calculate regime suitability score"""
        pattern_type = pattern_data.get('pattern_type', '')
        direction = pattern_data.get('direction', 'LONG')

        # Different patterns work better in different regimes
        if regime == MarketRegime.TRENDING_UP:
            if direction == 'LONG':
                return 80
            else:
                return 40
        elif regime == MarketRegime.TRENDING_DOWN:
            if direction == 'SHORT':
                return 80
            else:
                return 40
        elif regime == MarketRegime.RANGING:
            # Reversal patterns work better in ranging markets
            if 'Double' in pattern_type or 'Head' in pattern_type:
                return 75
            else:
                return 60
        else:  # VOLATILE
            # Breakout patterns work better in volatile markets
            if 'Breakout' in pattern_type:
                return 85
            else:
                return 50


class EnsemblePredictor:
    """Ensemble prediction combining multiple signals"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.ml_predictor = MLPredictor()
        self.confidence_scorer = EnhancedConfidenceScorer()

    def generate_ensemble_signal(self, df: pd.DataFrame, pattern_data: Dict) -> EnsembleSignal:
        """Generate ensemble signal combining all analysis methods"""
        try:
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(df)

            # 2. Extract ML features and predict
            features = self.ml_predictor.extract_features(
                df,
                pattern_data.get('confidence', 50),
                pattern_data.get('pattern_type', ''),
                pattern_data.get('direction', 'LONG')
            )
            ml_prediction = self.ml_predictor.predict(features)

            # 3. Calculate enhanced confidence
            enhanced_confidence = self.confidence_scorer.calculate_enhanced_confidence(
                df, pattern_data, regime, ml_prediction
            )

            # 4. Calculate component scores
            pattern_score = pattern_data.get('confidence', 50)
            regime_score = self._calculate_regime_score(regime, pattern_data)

            # 5. ML prediction score (convert to 0-100 scale)
            ml_score = (ml_prediction.confidence_score * 100)

            # 6. Risk adjustment based on volatility and regime
            risk_adjustment = self._calculate_risk_adjustment(df, regime, ml_prediction)

            # 7. Ensemble final score (weighted combination)
            weights = {
                'enhanced_confidence': 0.4,
                'ml_score': 0.3,
                'regime_score': 0.2,
                'risk_adjustment': 0.1
            }

            final_score = (
                enhanced_confidence * weights['enhanced_confidence'] +
                ml_score * weights['ml_score'] +
                regime_score * weights['regime_score'] +
                risk_adjustment * weights['risk_adjustment']
            )

            # Determine direction based on consensus
            direction = pattern_data.get('direction', 'LONG')
            if ml_prediction.predicted_return_24h < -0.01 and ml_prediction.confidence_score > 0.6:
                direction = 'SHORT'
            elif ml_prediction.predicted_return_24h > 0.01 and ml_prediction.confidence_score > 0.6:
                direction = 'LONG'

            return EnsembleSignal(
                direction=direction,
                confidence=enhanced_confidence,
                ml_prediction=ml_prediction.predicted_return_24h,
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
        pattern_type = pattern_data.get('pattern_type', '')
        direction = pattern_data.get('direction', 'LONG')

        if regime == MarketRegime.TRENDING_UP and direction == 'LONG':
            return 80
        elif regime == MarketRegime.TRENDING_DOWN and direction == 'SHORT':
            return 80
        elif regime == MarketRegime.RANGING:
            return 70
        elif regime == MarketRegime.VOLATILE and 'Breakout' in pattern_type:
            return 85
        else:
            return 50

    def _calculate_risk_adjustment(self, df: pd.DataFrame, regime: MarketRegime,
                                 ml_prediction: PredictionResult) -> float:
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

            # ML confidence adjustment
            if ml_prediction.confidence_score > 0.8:
                base_risk += 10  # High confidence = lower risk
            elif ml_prediction.confidence_score < 0.4:
                base_risk -= 10  # Low confidence = higher risk

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
        self.ml_predictor = MLPredictor()
        self.confidence_scorer = EnhancedConfidenceScorer()
        self.ensemble_predictor = EnsemblePredictor()
        self.position_sizer = RiskAdjustedPositionSizer()

    def analyze_pattern(self, df: pd.DataFrame, pattern_data: Dict) -> Dict[str, Any]:
        """Complete analysis of a pattern with all enhanced features"""
        try:
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(df)

            # 2. Generate ensemble signal
            ensemble_signal = self.ensemble_predictor.generate_ensemble_signal(df, pattern_data)

            # 3. Enhanced confidence scoring
            enhanced_confidence = self.confidence_scorer.calculate_enhanced_confidence(
                df, pattern_data, regime
            )

            # 4. ML prediction features
            features = self.ml_predictor.extract_features(
                df,
                pattern_data.get('confidence', 50),
                pattern_data.get('pattern_type', ''),
                pattern_data.get('direction', 'LONG')
            )
            ml_prediction = self.ml_predictor.predict(features)

            return {
                'regime': regime.value,
                'ensemble_signal': ensemble_signal,
                'enhanced_confidence': enhanced_confidence,
                'ml_prediction': ml_prediction,
                'features': features,
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
        if ensemble_signal.final_score > 70:
            return f"STRONG_{ensemble_signal.direction}"
        elif ensemble_signal.final_score > 55:
            return f"MODERATE_{ensemble_signal.direction}"
        elif ensemble_signal.final_score < 30:
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