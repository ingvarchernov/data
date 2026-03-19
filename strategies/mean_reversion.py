#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 MEAN REVERSION STRATEGY
Торгівля на відскоках від середніх значень
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Стратегія повернення до середнього:
    
    LONG умови:
    - Ціна нижче BB lower band
    - RSI < 30 (oversold)
    - Volume вище середнього (підтвердження)
    
    SHORT умови:
    - Ціна вище BB upper band
    - RSI > 70 (overbought)
    - Volume вище середнього (підтвердження)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'min_confidence': 60,
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_multiplier': 1.5,  # Volume має бути вище 1.5x середнього
            'sl_percent': 1.5,  # % SL від entry
            'tp_percent': 2.5,  # % TP від entry
        }
        if config:
            default_config.update(config)
        
        super().__init__("Mean Reversion", default_config)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахувати необхідні індикатори"""
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(self.config['bb_period']).mean()
        bb_std = df['close'].rolling(self.config['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.config['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.config['bb_std'])
        
        # BB Width (для визначення волатильності)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Відстань від BB bands
        df['dist_from_upper'] = (df['bb_upper'] - df['close']) / df['close'] * 100
        df['dist_from_lower'] = (df['close'] - df['bb_lower']) / df['close'] * 100
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Генерувати торговий сигнал"""
        # Розрахувати індикатори
        df = self.calculate_indicators(df)
        
        # Потрібно мінімум 20 свічок для BB
        if len(df) < 20:
            return None
        
        # Поточні значення
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Перевірка умов для LONG (ціна oversold)
        if current_price < bb_lower and rsi < self.config['rsi_oversold']:
            # Перевірка volume
            volume_confirmed = volume_ratio > self.config['volume_multiplier']
            
            # Розрахувати SL/TP
            stop_loss = current_price * (1 - self.config['sl_percent'] / 100)
            take_profit = bb_middle  # TP = повернення до середньої
            
            # Якщо TP занадто близько, використати % від entry
            if (take_profit - current_price) / current_price < 0.01:
                take_profit = current_price * (1 + self.config['tp_percent'] / 100)
            
            # Confidence
            confidence = 50
            confidence += (self.config['rsi_oversold'] - rsi)  # Більше балів якщо RSI нижче
            if volume_confirmed:
                confidence += 15
            confidence = min(95, confidence)
            
            return Signal(
                direction='LONG',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Mean Reversion LONG: Below BB ({current_price:.2f} < {bb_lower:.2f}), RSI={rsi:.1f}, Vol={'✅' if volume_confirmed else '❌'}",
                metadata={'rsi': rsi, 'bb_lower': bb_lower, 'volume_ratio': volume_ratio}
            )
        
        # Перевірка умов для SHORT (ціна overbought)
        elif current_price > bb_upper and rsi > self.config['rsi_overbought']:
            # Перевірка volume
            volume_confirmed = volume_ratio > self.config['volume_multiplier']
            
            # Розрахувати SL/TP
            stop_loss = current_price * (1 + self.config['sl_percent'] / 100)
            take_profit = bb_middle  # TP = повернення до середньої
            
            # Якщо TP занадто близько, використати % від entry
            if (current_price - take_profit) / current_price < 0.01:
                take_profit = current_price * (1 - self.config['tp_percent'] / 100)
            
            # Confidence
            confidence = 50
            confidence += (rsi - self.config['rsi_overbought'])  # Більше балів якщо RSI вище
            if volume_confirmed:
                confidence += 15
            confidence = min(95, confidence)
            
            return Signal(
                direction='SHORT',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Mean Reversion SHORT: Above BB ({current_price:.2f} > {bb_upper:.2f}), RSI={rsi:.1f}, Vol={'✅' if volume_confirmed else '❌'}",
                metadata={'rsi': rsi, 'bb_upper': bb_upper, 'volume_ratio': volume_ratio}
            )
        
        return None
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Розрахувати розмір позиції"""
        # Базовий розмір 2% від депозиту
        base_size = account_balance * 0.02
        
        # Збільшуємо розмір якщо confidence високий
        if signal.confidence > 75:
            base_size *= 1.4
        elif signal.confidence > 65:
            base_size *= 1.2
        
        # Обмежуємо максимальним розміром
        max_size = account_balance * 0.04
        return min(base_size, max_size)
