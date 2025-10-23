"""
Volume Indicators

Індикатори об'єму:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume SMA
- Volume ratios
"""

import logging
import asyncio
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PYTHON IMPLEMENTATIONS
# =============================================================================

def _pandas_obv(data: pd.DataFrame) -> pd.Series:
    """Python implementation for OBV"""
    obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
    return obv


def _pandas_vwap(data: pd.DataFrame) -> pd.Series:
    """Python implementation for VWAP"""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap


def _pandas_volume_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Python implementation for Volume SMA"""
    return data['volume'].rolling(window=period).mean()


def _pandas_volume_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    """Python implementation for Volume Ratio (current vs average)"""
    volume_sma = data['volume'].rolling(window=period).mean()
    ratio = data['volume'] / (volume_sma + 1e-10)
    return ratio


# =============================================================================
# ASYNC FUNCTIONS
# =============================================================================

async def calculate_obv_async(data: pd.DataFrame, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок OBV
    
    Args:
        data: DataFrame with OHLCV data
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with OBV values
    """
    return _pandas_obv(data)


async def calculate_vwap_async(data: pd.DataFrame, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок VWAP
    
    Args:
        data: DataFrame with OHLCV data
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with VWAP values
    """
    return _pandas_vwap(data)


async def calculate_volume_sma_async(data: pd.DataFrame, period: int = 20, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок Volume SMA
    
    Args:
        data: DataFrame with OHLCV data
        period: SMA period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with Volume SMA values
    """
    return _pandas_volume_sma(data, period)


async def calculate_volume_ratio_async(data: pd.DataFrame, period: int = 20, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок Volume Ratio
    
    Args:
        data: DataFrame with OHLCV data
        period: Period for average calculation
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with Volume Ratio values
    """
    return _pandas_volume_ratio(data, period)


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Синхронна обгортка для OBV
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        pd.Series with OBV values
    """
    result = _pandas_obv(data)
    result.name = 'OBV'
    return result


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Синхронна обгортка для VWAP
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        pd.Series with VWAP values
    """
    result = _pandas_vwap(data)
    result.name = 'VWAP'
    return result


def calculate_volume_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Синхронна обгортка для Volume SMA
    
    Args:
        data: DataFrame with OHLCV data
        period: SMA period
    
    Returns:
        pd.Series with Volume SMA values
    """
    result = _pandas_volume_sma(data, period)
    result.name = f'Volume_SMA_{period}'
    return result


def calculate_volume_ratio(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Синхронна обгортка для Volume Ratio
    
    Args:
        data: DataFrame with OHLCV data
        period: Period for average calculation
    
    Returns:
        pd.Series with Volume Ratio values
    """
    result = _pandas_volume_ratio(data, period)
    result.name = 'Volume_Ratio'
    return result
