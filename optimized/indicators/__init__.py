"""
Optimized Indicators Package

Модулі технічних індикаторів з підтримкою Rust acceleration:
- trend: Trend indicators (SMA, EMA, MACD)
- momentum: Momentum indicators (RSI, Stochastic, ROC)
- volatility: Volatility indicators (ATR, Bollinger Bands)
- volume: Volume indicators (OBV, Volume ratios)
- calculator: Main calculator class with async support
"""

import logging

logger = logging.getLogger(__name__)

# Перевірка доступності Rust модуля
try:
    import fast_indicators
    RUST_AVAILABLE = True
    logger.info("✅ Rust модуль fast_indicators завантажено")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"⚠️ Rust модуль недоступний: {e}. Використовуватиметься Python fallback")

from .calculator import OptimizedIndicatorCalculator, calculate_all_indicators
from .trend import (
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_trix
)
from .momentum import (
    calculate_rsi,
    calculate_stochastic,
    calculate_roc,
    calculate_momentum,
    calculate_williams_r,
    calculate_cci
)
from .volatility import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_adx
)
from .volume import (
    calculate_obv,
    calculate_vwap
)

__all__ = [
    'RUST_AVAILABLE',
    'OptimizedIndicatorCalculator',
    'calculate_all_indicators',
    # Trend
    'calculate_sma',
    'calculate_ema',
    'calculate_macd',
    'calculate_trix',
    # Momentum
    'calculate_rsi',
    'calculate_stochastic',
    'calculate_roc',
    'calculate_momentum',
    'calculate_williams_r',
    'calculate_cci',
    # Volatility
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_adx',
    # Volume
    'calculate_obv',
    'calculate_vwap',
]
