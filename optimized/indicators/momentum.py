"""
Momentum Indicators

Індикатори імпульсу з Rust acceleration:
- RSI (Relative Strength Index)
- Stochastic Oscillator
- ROC (Rate of Change)
- Momentum
- Williams %R
- CCI (Commodity Channel Index)
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

def _pandas_rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for RSI"""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()


def _pandas_stochastic(data: pd.DataFrame, k_period: int, smooth_k: int, smooth_d: int) -> Tuple[pd.Series, pd.Series]:
    """Python fallback for Stochastic"""
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low + 1e-10))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return k_smooth.dropna(), d_smooth.dropna()


def _pandas_roc(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for ROC"""
    roc = ((data['close'] - data['close'].shift(period)) / (data['close'].shift(period) + 1e-10)) * 100
    return roc.dropna()


def _pandas_momentum(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for Momentum"""
    momentum = data['close'] - data['close'].shift(period)
    return momentum.dropna()


def _pandas_williams_r(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for Williams %R"""
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low + 1e-10))
    return williams_r.dropna()


def _pandas_cci(data: pd.DataFrame, period: int) -> pd.Series:
    """Python fallback for CCI"""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci.dropna()


# =============================================================================
# ASYNC RUST-ACCELERATED FUNCTIONS
# =============================================================================

async def calculate_rsi_async(data: pd.DataFrame, period: int = 14, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок RSI
    
    Args:
        data: DataFrame with OHLCV data
        period: RSI period (default: 14)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with RSI values
    """
    if not RUST_AVAILABLE:
        return _pandas_rsi(data, period)
    
    try:
        prices = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            rsi_values = await loop.run_in_executor(
                executor,
                fast_indicators.fast_rsi,
                prices, period
            )
        else:
            rsi_values = fast_indicators.fast_rsi(prices, period)
        
        # RSI починається з period+1 позиції
        result_index = data.index[period:period+len(rsi_values)]
        return pd.Series(rsi_values, index=result_index, name=f'RSI_{period}')
        
    except Exception as e:
        logger.debug(f"Rust RSI failed, fallback to pandas: {e}")
        return _pandas_rsi(data, period)


async def calculate_stochastic_async(
    data: pd.DataFrame,
    k_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    executor=None
) -> Tuple[pd.Series, pd.Series]:
    """
    Асинхронний розрахунок Stochastic
    
    Args:
        data: DataFrame with OHLCV data
        k_period: %K period (default: 14)
        smooth_k: %K smoothing (default: 3)
        smooth_d: %D smoothing (default: 3)
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        Tuple of (%K, %D)
    """
    if not RUST_AVAILABLE:
        return _pandas_stochastic(data, k_period, smooth_k, smooth_d)
    
    try:
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            k, d = await loop.run_in_executor(
                executor,
                fast_indicators.fast_stochastic,
                high, low, close, k_period, smooth_k, smooth_d
            )
        else:
            k, d = fast_indicators.fast_stochastic(high, low, close, k_period, smooth_k, smooth_d)
        
        k_start = k_period - 1 + smooth_k - 1
        d_start = k_start + smooth_d - 1
        k_index = data.index[k_start:k_start+len(k)]
        d_index = data.index[d_start:d_start+len(d)]
        
        return (
            pd.Series(k, index=k_index, name='Stoch_K'),
            pd.Series(d, index=d_index, name='Stoch_D')
        )
        
    except Exception as e:
        logger.debug(f"Rust Stochastic failed, fallback to pandas: {e}")
        return _pandas_stochastic(data, k_period, smooth_k, smooth_d)


async def calculate_roc_async(data: pd.DataFrame, period: int = 10, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок ROC
    
    Args:
        data: DataFrame with OHLCV data
        period: ROC period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with ROC values
    """
    # ROC зазвичай не має Rust реалізації
    return _pandas_roc(data, period)


async def calculate_momentum_async(data: pd.DataFrame, period: int = 10, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок Momentum
    
    Args:
        data: DataFrame with OHLCV data
        period: Momentum period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with Momentum values
    """
    return _pandas_momentum(data, period)


async def calculate_williams_r_async(data: pd.DataFrame, period: int = 14, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок Williams %R
    
    Args:
        data: DataFrame with OHLCV data
        period: Williams %R period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with Williams %R values
    """
    return _pandas_williams_r(data, period)


async def calculate_cci_async(data: pd.DataFrame, period: int = 20, executor=None) -> pd.Series:
    """
    Асинхронний розрахунок CCI
    
    Args:
        data: DataFrame with OHLCV data
        period: CCI period
        executor: ThreadPoolExecutor for async execution
    
    Returns:
        pd.Series with CCI values
    """
    if not RUST_AVAILABLE:
        return _pandas_cci(data, period)
    
    try:
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        if executor:
            loop = asyncio.get_event_loop()
            cci_values = await loop.run_in_executor(
                executor,
                fast_indicators.fast_cci,
                high, low, close, period
            )
        else:
            cci_values = fast_indicators.fast_cci(high, low, close, period)
        
        result_index = data.index[period-1:period-1+len(cci_values)]
        return pd.Series(cci_values, index=result_index, name=f'CCI_{period}')
        
    except Exception as e:
        logger.debug(f"Rust CCI failed, fallback to pandas: {e}")
        return _pandas_cci(data, period)


# =============================================================================
# SYNCHRONOUS WRAPPERS
# =============================================================================

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Синхронна обгортка для RSI"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_rsi(data, period)
    except RuntimeError:
        return asyncio.run(calculate_rsi_async(data, period))


def calculate_stochastic(
    data: pd.DataFrame,
    k_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Синхронна обгортка для Stochastic"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_stochastic(data, k_period, smooth_k, smooth_d)
    except RuntimeError:
        return asyncio.run(calculate_stochastic_async(data, k_period, smooth_k, smooth_d))


def calculate_roc(data: pd.DataFrame, period: int = 10) -> pd.Series:
    """Синхронна обгортка для ROC"""
    return _pandas_roc(data, period)


def calculate_momentum(data: pd.DataFrame, period: int = 10) -> pd.Series:
    """Синхронна обгортка для Momentum"""
    return _pandas_momentum(data, period)


def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Синхронна обгортка для Williams %R"""
    return _pandas_williams_r(data, period)


def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Синхронна обгортка для CCI"""
    try:
        loop = asyncio.get_running_loop()
        return _pandas_cci(data, period)
    except RuntimeError:
        return asyncio.run(calculate_cci_async(data, period))
