"""
Rust-accelerated Feature Engineering
Використовує fast_indicators Rust бібліотеку для швидких розрахунків
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Rust indicators
try:
    import fast_indicators as fi
    RUST_AVAILABLE = True
    logger.info("✅ Rust indicators available")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("⚠️ Rust indicators not available, using Python fallback")


class RustFeatureEngineer:
    """
    Feature Engineer з Rust прискоренням
    
    Використовує Rust для всіх розрахунків індикаторів
    """
    
    def __init__(self, use_rust: bool = True):
        """
        Args:
            use_rust: Використовувати Rust (якщо доступно)
        """
        self.use_rust = use_rust and RUST_AVAILABLE
        
        if self.use_rust:
            logger.info("🦀 Using Rust-accelerated indicators")
        else:
            logger.info("🐍 Using Python indicators")
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        sma_periods: List[int] = [5, 10, 20, 50, 100, 200],
        ema_periods: List[int] = [9, 12, 21, 26, 50],
        rsi_periods: List[int] = [7, 14, 21, 28],
        atr_periods: List[int] = [14, 21],
    ) -> pd.DataFrame:
        """
        Розрахунок всіх індикаторів через Rust
        
        Args:
            df: OHLCV DataFrame
            sma_periods: Періоди для SMA
            ema_periods: Періоди для EMA
            rsi_periods: Періоди для RSI
            atr_periods: Періоди для ATR
        
        Returns:
            DataFrame з features
        """
        logger.info("📊 Розрахунок індикаторів через Rust...")
        result = df.copy()
        
        # Convert to numpy arrays with proper dtype
        close = result['close'].values.astype(np.float64)
        high = result['high'].values.astype(np.float64)
        low = result['low'].values.astype(np.float64)
        volume = result['volume'].values.astype(np.float64)
        
        # Basic features
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        if not self.use_rust:
            logger.error("❌ Rust not available!")
            return result
        
        # SMA indicators
        for period in sma_periods:
            result[f'sma_{period}'] = fi.sma(close, period)
            result[f'sma_{period}_ratio'] = close / result[f'sma_{period}'].values
            
        # EMA indicators
        for period in ema_periods:
            result[f'ema_{period}'] = fi.ema(close, period)
            result[f'ema_{period}_ratio'] = close / result[f'ema_{period}'].values
        
        # RSI indicators
        for period in rsi_periods:
            result[f'rsi_{period}'] = fi.rsi(close, period)
        
        # ATR indicators
        for period in atr_periods:
            result[f'atr_{period}'] = fi.atr(high, low, close, period)
        
        # Volume indicators
        result['obv'] = fi.obv(close, volume)
        result['vwap'] = fi.vwap(high, low, close, volume)
        
        # Volume features
        for period in [5, 10, 20, 30]:
            result[f'volume_mean_{period}'] = fi.sma(volume, period)
        
        result['volume_ratio'] = volume / result['volume_mean_20'].values
        result['volume_momentum'] = result['volume'].diff(5)
        
        # Volatility features
        for period in [5, 10, 20, 30]:
            result[f'close_std_{period}'] = fi.rolling_std(close, period)
            result[f'hvol_{period}'] = fi.historical_volatility(
                result['returns'].values, period
            )
        
        # Price features
        for period in [5, 10, 20]:
            sma_val = fi.sma(close, period)
            result[f'dist_from_mean_{period}'] = (close - sma_val) / sma_val
        
        # Momentum features
        result['price_momentum'] = result['close'].diff(10)
        result['acceleration'] = result['returns'].diff(5)
        
        # Candle features
        result['body'] = result['close'] - result['open']
        result['upper_wick'] = result['high'] - result[['open', 'close']].max(axis=1)
        result['lower_wick'] = result[['open', 'close']].min(axis=1) - result['low']
        result['body_ratio'] = abs(result['body']) / (result['high'] - result['low'])
        
        # Bollinger Bands indicators (simple implementation)
        for period in [20, 50]:
            sma_val = fi.sma(close, period)
            std_val = fi.rolling_std(close, period)
            
            result[f'bb_upper_{period}'] = sma_val + (2 * std_val)
            result[f'bb_lower_{period}'] = sma_val - (2 * std_val)
            result[f'bb_width_{period}'] = (
                (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']) / sma_val
            )
            result[f'bb_percent_{period}'] = (
                (close - result[f'bb_lower_{period}'].values) /
                (result[f'bb_upper_{period}'].values - result[f'bb_lower_{period}'].values + 1e-8)
            )
        
        # ROC
        for period in [5, 10, 20]:
            result[f'roc_{period}'] = ((close - np.roll(close, period)) / np.roll(close, period)) * 100
        
        # Additional features from training
        # Golden cross / Death cross
        result['golden_cross'] = (result['sma_50'] > result['sma_200']).astype(int) - (result['sma_50'] < result['sma_200']).astype(int)
        
        # MACD cross (simplified)
        ema_12 = fi.ema(close, 12)
        ema_26 = fi.ema(close, 26)
        macd = ema_12 - ema_26
        signal = fi.ema(macd, 9)
        result['macd_cross'] = (macd > signal).astype(int) - (macd < signal).astype(int)
        
        # Volume features
        result['volume_sma20'] = fi.sma(volume, 20)
        result['volume_trend'] = result['volume'].rolling(5).mean() / result['volume'].rolling(20).mean()
        
        # RSI signals
        result['rsi_overbought'] = (result['rsi_14'] > 70).astype(int)
        result['rsi_oversold'] = (result['rsi_14'] < 30).astype(int)
        
        # Momentum
        result['momentum_5'] = result['close'] / result['close'].shift(5) - 1
        result['momentum_10'] = result['close'] / result['close'].shift(10) - 1
        result['momentum_20'] = result['close'] / result['close'].shift(20) - 1
        
        # Volatility
        result['volatility_20'] = fi.historical_volatility(result['returns'].values, 20)
        
        # Price vs SMA ratios (additional)
        for period in [5, 10, 20, 50, 100, 200]:
            if f'sma_{period}' in result.columns:
                result[f'price_vs_sma{period}'] = close / result[f'sma_{period}'].values
        
        # Don't drop NaN - models can handle them
        # result = result.dropna()
        
        return result
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Отримати список feature names (виключаючи OHLCV)
        
        Args:
            df: DataFrame з features
        
        Returns:
            List feature names
        """
        exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                   'open_time', 'close_time']
        
        features = [col for col in df.columns if col not in exclude]
        return features


__all__ = ["RustFeatureEngineer", "RUST_AVAILABLE"]
