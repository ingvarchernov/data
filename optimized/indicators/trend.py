"""
Trend Indicators

Індикатори трендів з Rust acceleration:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- TRIX (Triple Exponential Average)
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

def _pandas_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for SMA"""
    return data['close'].rolling(window=period).mean()


def _pandas_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for EMA"""
    return data['close'].ewm(span=period, adjust=False).mean()


def _pandas_macd(data: pd.DataFrame, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Python fallback for MACD"""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _pandas_trix(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for TRIX"""
    ema1 = data['close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    return trix


# =============================================================================
# ASYNC RUST-ACCELERATED FUNCTIONS
# =============================================================================

async def calculate_sma_async(data: pd.DataFrame, period: int, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок SMA
    
    Args:
        data: DataFrame with OHLCV data
        period: SMA period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with SMA values
    """
    if not RUST_AVAILABLE:
        return _pandas_sma(data, period)
    
    try:
        prices = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            sma_values = await loop.run_in_executor(
                executor,
                fast_indicators.fast_sma,
                prices, period
            )
        else:
            # Синхронний виклик якщо немає executor
            sma_values = fast_indicators.fast_sma(prices, period)
        
        # Створюємо Series з правильним індексом
        result_index = data.index[period-1:period-1+len(sma_values)]
        return pd.Series(sma_values, index=result_index, name=f'SMA_{period}')
        
    except Exception as e:
        logger.debug(f"Rust SMA failed, fallback to pandas: {e}")
        return _pandas_sma(data, period)


async def calculate_ema_async(data: pd.DataFrame, period: int, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок EMA
    
    Args:
        data: DataFrame with OHLCV data
        period: EMA period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with EMA values
    """
    if not RUST_AVAILABLE:
        return _pandas_ema(data, period)
    
    try:
        prices = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            ema_values = await loop.run_in_executor(
                executor,
                fast_indicators.fast_ema,
                prices, period
            )
        else:
            ema_values = fast_indicators.fast_ema(prices, period)
        
        # EMA повертає повний масив
        return pd.Series(ema_values, index=data.index[:len(ema_values)], name=f'EMA_{period}')
        
    except Exception as e:
        logger.debug(f"Rust EMA failed, fallback to pandas: {e}")
        return _pandas_ema(data, period)


async def calculate_macd_async(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    executor=None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Асинхронний розрахунок MACD
    
    Args:
        data: DataFrame with OHLCV data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    if not RUST_AVAILABLE:
        return _pandas_macd(data, fast, slow, signal)
    
    try:
        prices = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            macd, signal_line, histogram = await loop.run_in_executor(
                executor,
                fast_indicators.fast_macd,
                prices, fast, slow, signal
            )
        else:
            macd, signal_line, histogram = fast_indicators.fast_macd(prices, fast, slow, signal)
        
        # Безпечний розрахунок індексів
        try:
            macd_start = max(slow - 1, 0)
            macd_end = min(macd_start + len(macd), len(data))
            macd_index = data.index[macd_start:macd_end]
            
            signal_start = max(macd_start + signal - 1, 0)
            signal_end = min(signal_start + len(signal_line), len(data))
            signal_index = data.index[signal_start:signal_end]
            
            hist_index = signal_index  # Histogram має ту ж довжину що і signal
            
            # Перевірка довжини
            if len(macd_index) != len(macd):
                logger.debug(f"MACD index mismatch, using pandas fallback")
                return _pandas_macd(data, fast, slow, signal)
            
            return (
                pd.Series(macd, index=macd_index, name='MACD'),
                pd.Series(signal_line, index=signal_index, name='MACD_Signal'),
                pd.Series(histogram, index=hist_index, name='MACD_Histogram')
            )
            
        except Exception as index_error:
            logger.debug(f"MACD index calculation error: {index_error}")
            return _pandas_macd(data, fast, slow, signal)
            
    except Exception as e:
        logger.debug(f"Rust MACD failed, fallback to pandas: {e}")
        return _pandas_macd(data, fast, slow, signal)


async def calculate_trix_async(data: pd.DataFrame, period: int = 15, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок TRIX
    
    Args:
        data: DataFrame with OHLCV data
        period: TRIX period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with TRIX values
    """
    # TRIX зазвичай не має Rust реалізації - використовуємо pandas
    return _pandas_trix(data, period)


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================

def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Синхронна обгортка для SMA"""
    try:
        loop = asyncio.get_running_loop()
        # Якщо є running loop - використовуємо fallback
        return _pandas_sma(data, period)
    except RuntimeError:
        return asyncio.run(calculate_sma_async(data, period))


def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Синхронна обгортка для EMA"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_ema(data, period)
    except RuntimeError:
        return asyncio.run(calculate_ema_async(data, period))


def calculate_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Синхронна обгортка для MACD"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_macd(data, fast, slow, signal)
    except RuntimeError:
        return asyncio.run(calculate_macd_async(data, fast, slow, signal))


def calculate_trix(data: pd.DataFrame, period: int = 15) -> pd.Series:
    """Синхронна обгортка для TRIX"""
    return _pandas_trix(data, period)
