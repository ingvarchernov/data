"""
🎯 ТОРГОВІ СТРАТЕГІЇ
Модульна система різних підходів до торгівлі
"""
from .base import BaseStrategy, Signal
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
]
