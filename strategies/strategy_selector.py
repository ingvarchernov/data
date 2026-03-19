#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 STRATEGY SELECTOR
Автоматичний вибір найкращої стратегії для поточних ринкових умов
"""
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

from strategies.base import BaseStrategy, Signal
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Вибір стратегії на основі поточних ринкових умов
    
    Логіка:
    - STRONG TREND → Trend Following
    - WEAK TREND / RANGE → Mean Reversion
    - HIGH VOLATILITY → адаптувати параметри
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Створити всі доступні стратегії
        self.strategies = {
            'trend_following': TrendFollowingStrategy(
                self.config.get('trend_following', {})
            ),
            'mean_reversion': MeanReversionStrategy(
                self.config.get('mean_reversion', {})
            ),
        }
        
        # Статистика performance
        self.performance = {name: {'wins': 0, 'losses': 0, 'total_pnl': 0.0} 
                           for name in self.strategies.keys()}
        
        logger.info(f"✅ Strategy Selector ініціалізовано з {len(self.strategies)} стратегій")
    
    def analyze_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Визначити поточний режим ринку"""
        # SMA для визначення тренду
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean() if len(df) >= 200 else None
        
        current_price = df['close'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        # Нахил SMA
        sma_slope = (df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]) / 10 if len(df) >= 10 else 0
        
        # Відстань від SMA
        distance_from_sma = abs(current_price - sma_50) / sma_50 * 100
        
        # Волатильність (стандартне відхилення returns)
        returns = df['close'].pct_change()
        volatility = returns.std() * 100
        
        # Визначення режиму
        if distance_from_sma > 5.0:  # Сильний тренд
            if current_price > sma_50 and sma_slope > 0:
                regime = 'STRONG_UPTREND'
            elif current_price < sma_50 and sma_slope < 0:
                regime = 'STRONG_DOWNTREND'
            else:
                regime = 'TREND_WEAKENING'
        elif distance_from_sma > 2.0:  # Помірний тренд
            if current_price > sma_50 and sma_slope > 0:
                regime = 'UPTREND'
            elif current_price < sma_50 and sma_slope < 0:
                regime = 'DOWNTREND'
            else:
                regime = 'RANGE'
        else:  # Флет
            regime = 'RANGE'
        
        # Визначення волатильності
        if volatility > 3.0:
            vol_regime = 'HIGH'
        elif volatility > 1.5:
            vol_regime = 'MEDIUM'
        else:
            vol_regime = 'LOW'
        
        return {
            'regime': regime,
            'volatility': vol_regime,
            'distance_from_sma': distance_from_sma,
            'sma_slope': sma_slope,
            'volatility_value': volatility
        }
    
    def select_strategy(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """
        Вибрати найкращу стратегію для поточних умов
        
        Returns:
            Назва стратегії або None
        """
        market_info = self.analyze_market_regime(df)
        regime = market_info['regime']
        
        logger.info(f"📊 {symbol}: Market regime = {regime}, Volatility = {market_info['volatility']}")
        
        # Вибір стратегії на основі режиму
        if regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND', 'UPTREND', 'DOWNTREND']:
            return 'trend_following'
        elif regime in ['RANGE', 'TREND_WEAKENING']:
            return 'mean_reversion'
        else:
            return None
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Згенерувати сигнал використовуючи найкращу стратегію
        
        Args:
            df: DataFrame з OHLCV даними
            symbol: Символ
            
        Returns:
            Signal або None
        """
        # Вибрати стратегію
        strategy_name = self.select_strategy(df, symbol)
        
        if not strategy_name:
            logger.warning(f"⚠️ {symbol}: Не вдалось вибрати стратегію")
            return None
        
        strategy = self.strategies[strategy_name]
        logger.info(f"🎯 {symbol}: Використовую {strategy.name}")
        
        # Згенерувати сигнал
        signal = strategy.generate_signal(df, symbol)
        
        if signal and strategy.validate_signal(signal):
            # Додати інформацію про стратегію
            if signal.metadata is None:
                signal.metadata = {}
            signal.metadata['strategy'] = strategy_name
            
            logger.info(f"✅ {symbol}: {strategy.name} згенерував сигнал: "
                       f"{signal.direction} ({signal.confidence:.1f}%)")
            return signal
        
        return None
    
    def get_all_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Отримати сигнали від всіх стратегій (для порівняння)
        
        Returns:
            Список валідних сигналів
        """
        signals = []
        
        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(df, symbol)
            if signal and strategy.validate_signal(signal):
                if signal.metadata is None:
                    signal.metadata = {}
                signal.metadata['strategy'] = name
                signals.append(signal)
        
        return signals
    
    def update_performance(self, strategy_name: str, pnl: float, is_win: bool):
        """Оновити статистику performance стратегії"""
        if strategy_name in self.performance:
            if is_win:
                self.performance[strategy_name]['wins'] += 1
            else:
                self.performance[strategy_name]['losses'] += 1
            self.performance[strategy_name]['total_pnl'] += pnl
    
    def get_best_strategy(self) -> Optional[str]:
        """Отримати найкращу стратегію за результатами"""
        best_strategy = None
        best_win_rate = 0
        
        for name, perf in self.performance.items():
            total = perf['wins'] + perf['losses']
            if total > 10:  # Мінімум 10 угод для оцінки
                win_rate = perf['wins'] / total
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_strategy = name
        
        return best_strategy
    
    def print_performance(self):
        """Вивести статистику performance"""
        logger.info("\n" + "="*80)
        logger.info("📊 PERFORMANCE СТРАТЕГІЙ")
        logger.info("="*80)
        
        for name, perf in self.performance.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                win_rate = perf['wins'] / total * 100
                avg_pnl = perf['total_pnl'] / total
                logger.info(f"{name:20s}: {perf['wins']:3d}W / {perf['losses']:3d}L "
                           f"({win_rate:5.1f}%) | Avg PnL: ${avg_pnl:6.2f}")
        
        logger.info("="*80)
