#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 TREND FOLLOWING STRATEGY
Торгівля в напрямку тренду з використанням SMA та RSI
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """
    Стратегія слідування за трендом:
    
    LONG умови:
    - Ціна вище SMA 50
    - SMA 50 зростає
    - RSI < 70 (не overbought)
    - Pullback до EMA 20
    
    SHORT умови:
    - Ціна нижче SMA 50
    - SMA 50 падає
    - RSI > 30 (не oversold)
    - Pullback до EMA 20
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'min_confidence': 65,
            'sma_period': 50,
            'ema_period': 20,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_multiplier_sl': 1.5,  # SL = ATR * multiplier
            'atr_multiplier_tp': 2.5,   # TP = ATR * multiplier
            'trend_strength_threshold': 2.0,  # % відстань від SMA
        }
        if config:
            default_config.update(config)
        
        super().__init__("Trend Following", default_config)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахувати необхідні індикатори"""
        # SMA
        df['sma_50'] = df['close'].rolling(self.config['sma_period']).mean()
        df['sma_200'] = df['close'].rolling(200).mean() if len(df) >= 200 else np.nan
        
        # EMA
        df['ema_20'] = df['close'].ewm(span=self.config['ema_period']).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Визначити поточний тренд"""
        current_price = df['close'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        # Нахил SMA (зростає чи падає)
        sma_slope = (df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]) / 10
        
        # Відстань від SMA
        distance_pct = (current_price - sma_50) / sma_50 * 100
        
        # Визначення тренду
        trend = "NEUTRAL"
        if current_price > sma_50 and sma_slope > 0:
            if abs(distance_pct) > self.config['trend_strength_threshold']:
                trend = "STRONG_UP"
            else:
                trend = "UP"
        elif current_price < sma_50 and sma_slope < 0:
            if abs(distance_pct) > self.config['trend_strength_threshold']:
                trend = "STRONG_DOWN"
            else:
                trend = "DOWN"
        
        return {
            'trend': trend,
            'distance_pct': distance_pct,
            'sma_slope': sma_slope,
            'sma_50': sma_50
        }
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Генерувати торговий сигнал"""
        # Розрахувати індикатори
        df = self.calculate_indicators(df)
        
        # Потрібно мінімум 50 свічок для SMA 50
        if len(df) < 50:
            return None
        
        # Поточні значення
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Визначити тренд
        trend_info = self.detect_trend(df)
        trend = trend_info['trend']
        
        # Перевірка умов для LONG
        if trend in ['UP', 'STRONG_UP']:
            # Ціна має бути біля EMA 20 (pullback)
            distance_from_ema = abs(current_price - ema_20) / ema_20 * 100
            
            # RSI не overbought
            if rsi < self.config['rsi_overbought'] and distance_from_ema < 2.0:
                # Розрахувати SL/TP на основі ATR
                stop_loss = current_price - (atr * self.config['atr_multiplier_sl'])
                take_profit = current_price + (atr * self.config['atr_multiplier_tp'])
                
                # Confidence базується на силі тренду та RSI
                confidence = 50 + (abs(trend_info['distance_pct']) * 5)
                confidence += (50 - rsi) / 2  # Більше балів якщо RSI нижче 50
                confidence = min(95, confidence)
                
                return Signal(
                    direction='LONG',
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Trend Following LONG: {trend}, RSI={rsi:.1f}, Distance from EMA={distance_from_ema:.2f}%",
                    metadata={'trend': trend, 'rsi': rsi, 'atr': atr}
                )
        
        # Перевірка умов для SHORT
        elif trend in ['DOWN', 'STRONG_DOWN']:
            # Ціна має бути біля EMA 20 (pullback)
            distance_from_ema = abs(current_price - ema_20) / ema_20 * 100
            
            # RSI не oversold
            if rsi > self.config['rsi_oversold'] and distance_from_ema < 2.0:
                # Розрахувати SL/TP на основі ATR
                stop_loss = current_price + (atr * self.config['atr_multiplier_sl'])
                take_profit = current_price - (atr * self.config['atr_multiplier_tp'])
                
                # Confidence базується на силі тренду та RSI
                confidence = 50 + (abs(trend_info['distance_pct']) * 5)
                confidence += (rsi - 50) / 2  # Більше балів якщо RSI вище 50
                confidence = min(95, confidence)
                
                return Signal(
                    direction='SHORT',
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Trend Following SHORT: {trend}, RSI={rsi:.1f}, Distance from EMA={distance_from_ema:.2f}%",
                    metadata={'trend': trend, 'rsi': rsi, 'atr': atr}
                )
        
        return None
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Розрахувати розмір позиції"""
        # Базовий розмір 2% від депозиту
        base_size = account_balance * 0.02
        
        # Збільшуємо розмір якщо confidence високий
        if signal.confidence > 80:
            base_size *= 1.5
        elif signal.confidence > 70:
            base_size *= 1.25
        
        # Обмежуємо максимальним розміром
        max_size = account_balance * 0.05
        return min(base_size, max_size)
