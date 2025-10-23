"""
Feature Engineer - Розрахунок та обробка технічних індикаторів

Консолідує всі індикатори в один клас:
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic, ROC
- Volatility: ATR, Bollinger Bands
- Volume: OBV, Volume ratios
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Інженерія features для ML моделей
    
    Використання:
        engineer = FeatureEngineer()
        df_with_features = engineer.calculate_all(df)
    """
    
    def __init__(self, include_advanced: bool = True):
        """
        Args:
            include_advanced: Включити розширені індикатори
        """
        self.include_advanced = include_advanced
    
    # ========================================================================
    # TREND INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        MACD Indicator
        
        Returns:
            macd, signal_line, histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3
    ) -> tuple:
        """
        Stochastic Oscillator
        
        Returns:
            %K, %D
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=3).mean()
        
        return k, d
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def calculate_momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Momentum"""
        return data - data.shift(period)
    
    # ========================================================================
    # VOLATILITY INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """
        Bollinger Bands
        
        Returns:
            upper, middle, lower, width, %b
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle
        percent_b = (data - lower) / (upper - lower)
        
        return upper, middle, lower, width, percent_b
    
    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()
    
    # ========================================================================
    # PRICE PATTERNS
    # ========================================================================
    
    @staticmethod
    def calculate_candle_features(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """Свічкові патерни"""
        features = pd.DataFrame(index=close.index)
        
        features['body'] = close - open_
        features['upper_wick'] = high - pd.concat([open_, close], axis=1).max(axis=1)
        features['lower_wick'] = pd.concat([open_, close], axis=1).min(axis=1) - low
        features['body_ratio'] = abs(features['body']) / (high - low)
        
        return features
    
    # ========================================================================
    # MAIN METHOD
    # ========================================================================
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        sma_periods: List[int] = [7, 14, 21, 50, 100, 200],
        ema_periods: List[int] = [7, 14, 21, 50],
        rsi_periods: List[int] = [7, 14, 21],
        atr_periods: List[int] = [7, 14, 21]
    ) -> pd.DataFrame:
        """
        Розрахунок всіх індикаторів
        
        Args:
            df: OHLCV DataFrame
            sma_periods: Періоди для SMA
            ema_periods: Періоди для EMA
            rsi_periods: Періоди для RSI
            atr_periods: Періоди для ATR
        
        Returns:
            DataFrame з features
        """
        logger.info("📊 Розрахунок технічних індикаторів...")
        result = df.copy()
        
        # Basic features
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Trend indicators
        for period in sma_periods:
            result[f'sma_{period}'] = self.calculate_sma(result['close'], period)
            result[f'sma_{period}_ratio'] = result['close'] / result[f'sma_{period}']
        
        for period in ema_periods:
            result[f'ema_{period}'] = self.calculate_ema(result['close'], period)
            result[f'ema_{period}_ratio'] = result['close'] / result[f'ema_{period}']
        
        # MACD
        macd, signal, hist = self.calculate_macd(result['close'])
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = hist
        
        # Momentum indicators
        for period in rsi_periods:
            result[f'rsi_{period}'] = self.calculate_rsi(result['close'], period)
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(
            result['high'], result['low'], result['close']
        )
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d
        
        # ROC
        for period in [5, 10, 20]:
            result[f'roc_{period}'] = self.calculate_roc(result['close'], period)
        
        # Momentum
        for period in [5, 10, 20]:
            result[f'momentum_{period}'] = self.calculate_momentum(result['close'], period)
        
        # Volatility indicators
        for period in atr_periods:
            result[f'atr_{period}'] = self.calculate_atr(
                result['high'], result['low'], result['close'], period
            )
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower, width, percent_b = self.calculate_bollinger_bands(
                result['close'], period
            )
            result[f'bb_upper_{period}'] = upper
            result[f'bb_middle_{period}'] = middle
            result[f'bb_lower_{period}'] = lower
            result[f'bb_width_{period}'] = width
            result[f'bb_percent_{period}'] = percent_b
        
        # Volume indicators
        result['obv'] = self.calculate_obv(result['close'], result['volume'])
        result['volume_sma_20'] = self.calculate_volume_sma(result['volume'], 20)
        result['volume_ratio'] = result['volume'] / result['volume_sma_20']
        
        # Candle features
        candle_features = self.calculate_candle_features(
            result['open'], result['high'], result['low'], result['close']
        )
        result = pd.concat([result, candle_features], axis=1)
        
        # Advanced features
        if self.include_advanced:
            # Price distance from MAs
            for period in [10, 20, 50]:
                result[f'dist_from_sma_{period}'] = (
                    (result['close'] - result[f'sma_{period}']) / result[f'sma_{period}']
                )
            
            # Volatility
            for period in [7, 14, 20, 30]:
                result[f'volatility_{period}'] = result['returns'].rolling(period).std()
            
            # Rolling statistics
            for period in [5, 10, 20]:
                result[f'close_std_{period}'] = result['close'].rolling(period).std()
                result[f'close_mean_{period}'] = result['close'].rolling(period).mean()
        
        # Видалення NaN
        features_count_before = len(result.columns)
        result = result.dropna()
        features_count = len(result.columns)
        
        logger.info(f"✅ Розраховано {features_count} features ({len(result)} записів)")
        
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
