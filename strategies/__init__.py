"""
Trading Strategies Package
"""
from .base_strategy import BaseStrategy, Signal
from .trend_strategy_4h import TrendStrategy4h
from .swing_strategy_1h import SwingStrategy1h

__all__ = ['BaseStrategy', 'Signal', 'TrendStrategy4h', 'SwingStrategy1h']
