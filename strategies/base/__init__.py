"""
Базові класи та інтерфейси для торгових стратегій
"""
import abc
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import logging

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
    """Сигнал для торгівлі"""
    action: TradeAction
    symbol: str
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Відкрита позиція"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def current_value(self) -> float:
        """Поточна вартість позиції"""
        # Це буде розраховуватися в торговому боті
        return self.entry_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Нереалізований P&L"""
        # Це буде розраховуватися в торговому боті
        return 0.0


class TradingStrategy(abc.ABC):
    """Базовий клас для торгових стратегій"""

    def __init__(self, name: str, symbols: List[str], config: Optional[Dict] = None):
        self.name = name
        self.symbols = symbols
        self.config = config or {}
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Налаштування за замовчуванням
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1%
        self.max_drawdown = self.config.get('max_drawdown', 0.05)  # 5%

    @abc.abstractmethod
    def analyze_market(self, market_data: Dict[str, pd.DataFrame],
                      predictions: Dict[str, Dict]) -> Dict[str, TradeSignal]:
        """
        Аналіз ринку та генерація сигналів

        Args:
            market_data: Словник з OHLCV даними по символах
            predictions: Прогнози моделі по символах

        Returns:
            Словник сигналів по символам
        """
        pass

    @abc.abstractmethod
    def should_enter_position(self, symbol: str, signal: TradeSignal,
                            current_positions: Dict[str, Position]) -> bool:
        """Перевірка умов входу в позицію"""
        pass

    @abc.abstractmethod
    def should_exit_position(self, symbol: str, position: Position,
                           current_price: float, market_data: pd.DataFrame) -> bool:
        """Перевірка умов виходу з позиції"""
        pass

    def calculate_position_size(self, capital: float, entry_price: float,
                              stop_loss: float) -> float:
        """Розрахунок розміру позиції на основі ризику"""
        risk_amount = capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance == 0:
            return 0

        position_size = risk_amount / stop_distance
        max_position_size = capital * 0.02  # Максимум 2% від капіталу

        return min(position_size, max_position_size)

    def update_positions(self, signals: Dict[str, TradeSignal],
                        current_prices: Dict[str, float]) -> List[TradeSignal]:
        """Оновлення позицій на основі сигналів"""
        actions = []

        for symbol, signal in signals.items():
            current_price = current_prices.get(symbol, 0)

            # Перевірка умов входу
            if signal.action in [TradeAction.BUY, TradeAction.SELL]:
                if self.should_enter_position(symbol, signal, self.positions):
                    # Створюємо позицію
                    position = self._create_position(signal, current_price)
                    self.positions[symbol] = position
                    actions.append(signal)
                    self.logger.info(f"📈 Відкрита позиція {symbol}: {signal.action.value} at {current_price}")

            # Перевірка умов виходу
            elif signal.action == TradeAction.CLOSE and symbol in self.positions:
                position = self.positions[symbol]
                if self.should_exit_position(symbol, position, current_price, None):
                    # Закриваємо позицію
                    close_signal = TradeSignal(
                        action=TradeAction.CLOSE,
                        symbol=symbol,
                        confidence=1.0,
                        entry_price=current_price,
                        metadata={'position': position}
                    )
                    del self.positions[symbol]
                    actions.append(close_signal)
                    self.logger.info(f"📉 Закрита позиція {symbol} at {current_price}")

        return actions

    def _create_position(self, signal: TradeSignal, current_price: float) -> Position:
        """Створення нової позиції"""
        side = 'LONG' if signal.action == TradeAction.BUY else 'SHORT'

        return Position(
            symbol=signal.symbol,
            side=side,
            entry_price=signal.entry_price or current_price,
            quantity=signal.quantity or 0.001,  # За замовчуванням
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata={'strategy': self.name, 'signal_confidence': signal.confidence}
        )

    def get_open_positions(self) -> Dict[str, Position]:
        """Отримання відкритих позицій"""
        return self.positions.copy()

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Статистика стратегії"""
        return {
            'name': self.name,
            'open_positions': len(self.positions),
            'max_positions': self.max_positions,
            'risk_per_trade': self.risk_per_trade,
            'symbols': self.symbols
        }


class RiskManager:
    """Менеджер ризиків"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5%
        self.max_drawdown = self.config.get('max_drawdown', 0.10)  # 10%
        self.max_positions = self.config.get('max_positions', 5)

        # Стан
        self.daily_pnl = 0.0
        self.peak_capital = 0.0
        self.current_capital = 0.0

    def can_open_position(self, capital: float, position_size: float,
                         current_positions: int) -> Tuple[bool, str]:
        """Перевірка можливості відкриття позиції"""

        # Перевірка кількості позицій
        if current_positions >= self.max_positions:
            return False, f"Досягнуто максимальну кількість позицій ({self.max_positions})"

        # Перевірка розміру позиції
        if position_size > capital * 0.05:  # Максимум 5% від капіталу
            return False, f"Розмір позиції занадто великий ({position_size/capital:.1%})"

        # Перевірка денного збитку
        if self.daily_pnl < -self.max_daily_loss * capital:
            return False, f"Досягнуто денний ліміт збитку ({self.daily_pnl/capital:.1%})"

        # Перевірка drawdown
        if capital < self.peak_capital * (1 - self.max_drawdown):
            return False, f"Досягнуто максимальний drawdown ({self.max_drawdown:.1%})"

        return True, "OK"

    def update_capital(self, new_capital: float):
        """Оновлення капіталу"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

    def reset_daily_pnl(self):
        """Скидання денного P&L"""
        self.daily_pnl = 0.0