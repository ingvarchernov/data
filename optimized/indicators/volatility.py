"""
Volatility Indicators

Індикатори волатильності з Rust acceleration:
- ATR (Average True Range)
- Bollinger Bands
- ADX (Average Directional Index)
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

# Import Rust module if available
try:
    import fast_indicators
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


# =============================================================================
# PYTHON FALLBACKS
# =============================================================================

def _pandas_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for ATR"""
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.dropna()


def _pandas_bollinger(data: pd.DataFrame, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Python fallback for Bollinger Bands"""
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.dropna(), sma.dropna(), lower.dropna()


def _pandas_adx(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for ADX"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high.diff()
    dm_minus = low.diff() * -1
    
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    
    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    di_plus = (dm_plus.rolling(window=period).mean() / (atr + 1e-10)) * 100
    di_minus = (dm_minus.rolling(window=period).mean() / (atr + 1e-10)) * 100
    
    # ADX calculation
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)) * 100
    adx = dx.rolling(window=period).mean()
    
    return adx.dropna()


# =============================================================================
# ASYNC RUST-ACCELERATED FUNCTIONS
# =============================================================================

async def calculate_atr_async(data: pd.DataFrame, period: int = 14, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок ATR
    
    Args:
        data: DataFrame with OHLCV data
        period: ATR period (default: 14)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with ATR values
    """
    if not RUST_AVAILABLE:
        return _pandas_atr(data, period)
    
    try:
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            atr_values = await loop.run_in_executor(
                executor,
                fast_indicators.fast_atr,
                high, low, close, period
            )
        else:
            atr_values = fast_indicators.fast_atr(high, low, close, period)
        
        result_index = data.index[period:period+len(atr_values)]
        return pd.Series(atr_values, index=result_index, name=f'ATR_{period}')
        
    except Exception as e:
        logger.debug(f"Rust ATR failed, fallback to pandas: {e}")
        return _pandas_atr(data, period)


async def calculate_bollinger_bands_async(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    executor=None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Асинхронний розрахунок Bollinger Bands
    
    Args:
        data: DataFrame with OHLCV data
        period: Period for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        Tuple of (Upper Band, Middle Band/SMA, Lower Band)
    """
    if not RUST_AVAILABLE:
        return _pandas_bollinger(data, period, std_dev)
    
    try:
        prices = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            upper, lower = await loop.run_in_executor(
                executor,
                fast_indicators.fast_bollinger_bands,
                prices, period, std_dev
            )
        else:
            upper, lower = fast_indicators.fast_bollinger_bands(prices, period, std_dev)
        
        # Розраховуємо middle band (SMA)
        sma = prices[period-1:period-1+len(upper)]
        middle = (upper + lower) / 2
        
        result_index = data.index[period-1:period-1+len(upper)]
        
        return (
            pd.Series(upper, index=result_index, name='BB_Upper'),
            pd.Series(middle, index=result_index, name='BB_Middle'),
            pd.Series(lower, index=result_index, name='BB_Lower')
        )
        
    except Exception as e:
        logger.debug(f"Rust Bollinger Bands failed, fallback to pandas: {e}")
        return _pandas_bollinger(data, period, std_dev)


async def calculate_adx_async(data: pd.DataFrame, period: int = 14, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок ADX
    
    Args:
        data: DataFrame with OHLCV data
        period: ADX period (default: 14)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with ADX values
    """
    # ADX зазвичай не має Rust реалізації - використовуємо pandas
    return _pandas_adx(data, period)


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Синхронна обгортка для ATR"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_atr(data, period)
    except RuntimeError:
        return asyncio.run(calculate_atr_async(data, period))


def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Синхронна обгортка для Bollinger Bands"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_bollinger(data, period, std_dev)
    except RuntimeError:
        return asyncio.run(calculate_bollinger_bands_async(data, period, std_dev))


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Синхронна обгортка для ADX"""
    return _pandas_adx(data, period)
