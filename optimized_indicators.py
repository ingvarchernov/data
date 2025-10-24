"""
Backward compatibility wrapper for optimized indicators modules

This file provides backward compatibility for imports from 'optimized_indicators'.
New code should use: from optimized.indicators import ...
"""
import warnings

warnings.warn(
    "Importing from 'optimized_indicators' is deprecated. "
    "Use 'from optimized.indicators import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import all indicator components
from optimized.indicators import (
    RUST_AVAILABLE,
    OptimizedIndicatorCalculator,
    calculate_all_indicators,
    # Trend
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_trix,
    # Momentum
    calculate_rsi,
    calculate_stochastic,
    calculate_roc,
    calculate_momentum,
    calculate_williams_r,
    calculate_cci,
    # Volatility
    calculate_atr,
    calculate_bollinger_bands,
    calculate_adx,
    # Volume
    calculate_obv,
    calculate_vwap,
)

__all__ = [
    'RUST_AVAILABLE',
    'OptimizedIndicatorCalculator',
    'calculate_all_indicators',
    'calculate_sma',
    'calculate_ema',
    'calculate_macd',
    'calculate_trix',
    'calculate_rsi',
    'calculate_stochastic',
    'calculate_roc',
    'calculate_momentum',
    'calculate_williams_r',
    'calculate_cci',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_adx',
    'calculate_obv',
    'calculate_vwap',
]
