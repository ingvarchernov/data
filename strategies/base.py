#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📐 БАЗОВА СТРАТЕГІЯ
Абстрактний клас для всіх торгових стратегій
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class Signal:
    """Торговий сигнал"""
    direction: str  # 'LONG', 'SHORT', or 'NEUTRAL'
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str  # Пояснення чому відкриваємо
    metadata: Dict[str, Any] = None  # Додаткова інформація
    
    def is_valid(self) -> bool:
        """Перевірити чи сигнал валідний"""
        if self.direction not in ['LONG', 'SHORT', 'NEUTRAL']:
            return False
        if not (0 <= self.confidence <= 100):
            return False
        if self.entry_price <= 0:
            return False
        if self.stop_loss <= 0 or self.take_profit <= 0:
            return False
        return True


class BaseStrategy(ABC):
    """Базовий клас для всіх стратегій"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 60)
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Генерувати торговий сигнал на основі даних
        
        Args:
            df: DataFrame з OHLCV даними та індикаторами
            symbol: Символ (BTCUSDT, ETHUSDT, etc.)
            
        Returns:
            Signal або None якщо немає сигналу
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Розрахувати розмір позиції
        
        Args:
            signal: Торговий сигнал
            account_balance: Баланс рахунку
            
        Returns:
            Розмір позиції в USDT
        """
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """Валідувати сигнал перед відкриттям позиції"""
        if not signal or not signal.is_valid():
            return False
        
        if signal.confidence < self.min_confidence:
            return False
        
        if signal.direction == 'NEUTRAL':
            return False
        
        # Перевірка SL/TP
        if signal.direction == 'LONG':
            if signal.stop_loss >= signal.entry_price:
                return False
            if signal.take_profit <= signal.entry_price:
                return False
        else:  # SHORT
            if signal.stop_loss <= signal.entry_price:
                return False
            if signal.take_profit >= signal.entry_price:
                return False
        
        return True
    
    def __str__(self):
        return f"{self.name} Strategy"
