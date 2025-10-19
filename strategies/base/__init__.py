"""
Базові класи та інтерфейси для торгових стратегій
"""
import abc
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Дії для торгівлі"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class OrderType(Enum):
    """Типи ордерів"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class TradeSignal:
    """Торговий сигнал"""
    symbol: str
    action: TradeAction
    confidence: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник"""
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Position:
    """Відкрита позиція"""
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_pnl(self, current_price: float):
        """Оновлення нереалізованого P&L"""
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy_name': self.strategy_name,
            'unrealized_pnl': self.unrealized_pnl,
            'metadata': self.metadata
        }


class TradingStrategy(abc.ABC):
    """Базовий клас для торгових стратегій"""

    def __init__(self, name: str, symbols: List[str], config: Optional[Dict] = None):
        self.name = name
        self.symbols = symbols
        self.config = config or {}
        self.positions: Dict[str, Position] = {}
        self.active = True
        self.max_position_size = config.get('max_position_size', 0.1) if config else 0.1  # 10% за замовчуванням
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        logger.info(f"✅ Стратегія {name} ініціалізована для {len(symbols)} символів")

    @abc.abstractmethod
    async def analyze_market(
        self, 
        market_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, Dict]
    ) -> Dict[str, TradeSignal]:
        """
        Аналіз ринку та генерація сигналів
        
        Args:
            market_data: Історичні дані по символах
            predictions: ML прогнози по символах
            
        Returns:
            Dict[symbol, TradeSignal]: Сигнали для кожного символу
        """
        pass

    @abc.abstractmethod
    async def should_close_position(
        self, 
        position: Position, 
        current_price: float,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Перевірка чи потрібно закривати позицію
        
        Args:
            position: Відкрита позиція
            current_price: Поточна ціна
            market_data: Поточні дані ринку
            
        Returns:
            bool: True якщо треба закривати
        """
        pass

    def update_performance(self, pnl: float):
        """Оновлення статистики виконання"""
        self.performance['total_trades'] += 1
        self.performance['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = (
                self.performance['winning_trades'] / self.performance['total_trades']
            )

    def get_performance(self) -> Dict[str, Any]:
        """Отримання статистики"""
        return self.performance.copy()

    def add_position(self, position: Position):
        """Додавання позиції"""
        self.positions[position.symbol] = position
        logger.info(f"📊 {self.name}: відкрита позиція {position.symbol} {position.side}")

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Видалення позиції"""
        position = self.positions.pop(symbol, None)
        if position:
            logger.info(f"📊 {self.name}: закрита позиція {symbol}")
        return position

    def has_position(self, symbol: str) -> bool:
        """Перевірка наявності позиції"""
        return symbol in self.positions


class RiskManager:
    """Менеджер ризиків для стратегій"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% капіталу
        self.max_leverage = self.config.get('max_leverage', 3)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5% від капіталу
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.max_daily_trades = self.config.get('max_daily_trades', 50)
        
        logger.info(f"✅ RiskManager: max_pos={self.max_position_size:.1%}, leverage={self.max_leverage}")
    
    def validate_trade(
        self, 
        signal: TradeSignal, 
        portfolio_value: float,
        current_positions: int
    ) -> Tuple[bool, str]:
        """
        Валідація торгового сигналу
        
        Returns:
            (is_valid, reason)
        """
        # Перевірка daily loss
        if abs(self.daily_loss) >= self.max_daily_loss * portfolio_value:
            return False, f"Досягнуто daily loss limit: {self.daily_loss:.2f}"
        
        # Перевірка кількості угод
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Досягнуто daily trades limit: {self.daily_trades}"
        
        # Перевірка розміру позиції
        max_position_value = portfolio_value * self.max_position_size
        signal_value = signal.entry_price * signal.quantity
        if signal_value > max_position_value:
            return False, f"Позиція занадто велика: {signal_value:.2f} > {max_position_value:.2f}"
        
        # Перевірка confidence
        min_confidence = self.config.get('min_confidence', 0.6)
        if signal.confidence < min_confidence:
            return False, f"Низька впевненість: {signal.confidence:.2f} < {min_confidence}"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Розрахунок розміру позиції на основі ризику
        
        Args:
            signal: Торговий сигнал
            portfolio_value: Вартість портфеля
            risk_per_trade: Ризик на угоду (2% за замовчуванням)
            
        Returns:
            Розмір позиції в базовій валюті
        """
        if not signal.stop_loss:
            # Без stop-loss використовуємо фіксований розмір
            return portfolio_value * self.max_position_size / signal.entry_price
        
        # Розрахунок з урахуванням stop-loss
        risk_amount = portfolio_value * risk_per_trade
        price_risk = abs(signal.entry_price - signal.stop_loss)
        
        if price_risk > 0:
            quantity = risk_amount / price_risk
            max_quantity = (portfolio_value * self.max_position_size) / signal.entry_price
            return min(quantity, max_quantity)
        
        return portfolio_value * self.max_position_size / signal.entry_price
    
    def update_daily_stats(self, pnl: float):
        """Оновлення денної статистики"""
        self.daily_loss += pnl
        self.daily_trades += 1
    
    def reset_daily_stats(self):
        """Скидання денної статистики"""
        self.daily_loss = 0.0
        self.daily_trades = 0
        logger.info("🔄 RiskManager: daily stats reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Отримання метрик ризику"""
        return {
            'daily_loss': self.daily_loss,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss': self.max_daily_loss,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage
        }


# Експорт класів
__all__ = [
    'TradeAction',
    'OrderType',
    'TradeSignal',
    'Position',
    'TradingStrategy',
    'RiskManager'
]