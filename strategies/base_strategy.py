"""
Базовий клас для торгових стратегій
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Торговий сигнал"""
    symbol: str
    strategy: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Базовий клас стратегії"""
    
    def __init__(
        self,
        name: str,
        timeframe: str,
        symbols: List[str],
        min_confidence: float = 0.70,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05
    ):
        self.name = name
        self.timeframe = timeframe
        self.symbols = symbols
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Моделі та scaler'и per-symbol
        self.models = {}
        self.scalers = {}
        self.feature_names_dict = {}
        
        # Статистика
        self.total_signals = 0
        self.successful_signals = 0
        
    @abstractmethod
    async def load_models(self):
        """Завантаження ML моделей"""
        pass
    
    @abstractmethod
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Генерація торгового сигналу"""
        pass
    
    @abstractmethod
    async def should_close(self, symbol: str, position: Dict, current_price: float) -> tuple[bool, str]:
        """Чи треба закривати позицію"""
        pass
    
    def get_interval_seconds(self) -> int:
        """Інтервал перевірки в секундах"""
        intervals = {
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return intervals.get(self.timeframe, 3600)
    
    def get_stats(self) -> Dict:
        """Статистика стратегії"""
        win_rate = (self.successful_signals / self.total_signals * 100) if self.total_signals > 0 else 0
        return {
            'name': self.name,
            'timeframe': self.timeframe,
            'symbols': len(self.symbols),
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'win_rate': win_rate
        }
    
    def __str__(self):
        return f"{self.name} [{self.timeframe}] ({len(self.symbols)} symbols)"
