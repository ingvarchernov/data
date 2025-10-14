"""
Інтеграція з Binance Testnet для тестування торгових стратегій
"""
import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import time

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    try:
        from binance.websockets import BinanceSocketManager
        WEBSOCKET_AVAILABLE = True
    except ImportError:
        BinanceSocketManager = None
        WEBSOCKET_AVAILABLE = False
    BINANCE_AVAILABLE = True
except ImportError as e:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception
    BinanceRequestException = Exception
    BinanceSocketManager = None
    WEBSOCKET_AVAILABLE = False
    logging.warning(f"python-binance не встановлено або помилка імпорту: {e}. Встановіть: pip install python-binance")

import pandas as pd

from strategies.base import TradingStrategy, TradeSignal, TradeAction, Position, RiskManager

logger = logging.getLogger(__name__)


class BinanceTestnetTrader:
    """
    Трейдер для тестування стратегій на Binance Testnet

    Особливості:
    - Тестові кошти (без реального ризику)
    - Реальний час та дані
    - Симуляція комісій та slippage
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 test_mode: bool = True):
        """
        Ініціалізація трейдера

        Args:
            api_key: Binance API ключ
            api_secret: Binance API секрет
            test_mode: Використовувати testnet
        """
        self.test_mode = test_mode

        # API ключі (з змінних середовища або параметрів)
        self.api_key = api_key or os.getenv('BINANCE_TEST_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_TEST_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError("Потрібно вказати BINANCE_TEST_API_KEY та BINANCE_TEST_API_SECRET")

        # Ініціалізація клієнта
        self.client = Client(self.api_key, self.api_secret, testnet=test_mode)

        # Стан рахунку
        self.balance = {}  # Баланс по активах
        self.positions = {}  # Відкриті позиції
        self.order_history = []  # Історія ордерів

        # Налаштування
        self.commission_rate = 0.001  # 0.1% комісія
        self.slippage = 0.0005  # 0.05% slippage
        self.min_order_size = 10  # Мінімальний розмір ордера в USDT

        # WebSocket менеджер для real-time даних
        self.socket_manager = None
        self.streams = {}

        # Ініціалізація
        self._initialize_account()

        logger.info(f"✅ Binance {'Testnet' if test_mode else 'Live'} трейдер ініціалізований")

    def _initialize_account(self):
        """Ініціалізація рахунку"""
        try:
            account_info = self.client.get_account()

            # Отримання балансу
            for balance in account_info['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])

                if free + locked > 0:
                    self.balance[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }

            logger.info(f"💰 Баланс рахунку: {self.balance}")

        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації рахунку: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Отримання інформації про торговий символ"""
        try:
            info = self.client.get_symbol_info(symbol)
            return {
                'symbol': info['symbol'],
                'base_asset': info['baseAsset'],
                'quote_asset': info['quoteAsset'],
                'min_qty': float(info['filters'][1]['minQty']),
                'max_qty': float(info['filters'][1]['maxQty']),
                'step_size': float(info['filters'][1]['stepSize']),
                'min_price': float(info['filters'][0]['minPrice']),
                'max_price': float(info['filters'][0]['maxPrice']),
                'tick_size': float(info['filters'][0]['tickSize'])
            }
        except Exception as e:
            logger.error(f"❌ Помилка отримання інформації про {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Отримання поточної ціни"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"❌ Помилка отримання ціни {symbol}: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Отримання останніх угод"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return [{
                'price': float(trade['price']),
                'qty': float(trade['qty']),
                'time': datetime.fromtimestamp(trade['time'] / 1000),
                'is_buyer_maker': trade['isBuyerMaker']
            } for trade in trades]
        except Exception as e:
            logger.error(f"❌ Помилка отримання угод {symbol}: {e}")
            return []

    def place_market_order(self, symbol: str, side: str, quantity: float,
                          test: bool = True) -> Optional[Dict]:
        """
        Розміщення маркет ордера

        Args:
            symbol: Торговий символ
            side: 'BUY' або 'SELL'
            quantity: Кількість
            test: Тестовий ордер (без виконання)
        """
        try:
            # Отримання інформації про символ
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None

            # Округлення кількості до step_size
            step_size = symbol_info['step_size']
            quantity = self._round_to_step(quantity, step_size)

            # Форматування кількості як рядок без зайвих нулів
            quantity_str = f"{quantity:.8f}".rstrip('0').rstrip('.')
            logger.info(f"Форматована кількість для ордера: '{quantity_str}'")

            # Мінімальна перевірка кількості
            if quantity < symbol_info['min_qty']:
                logger.warning(f"Кількість {quantity} менше мінімальної {symbol_info['min_qty']}")
                return None

            # Перевірка балансу
            if not self._check_balance(symbol, side, quantity):
                return None

            # Розміщення ордера
            if test:
                # Тестовий ордер
                order = self.client.create_test_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity_str
                )
                order_id = f"test_{int(time.time())}"
            else:
                # Реальний ордер
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity_str
                )
                order_id = order['orderId']

            # Симуляція виконання (для тестування)
            executed_price = self._simulate_execution(symbol, side, quantity)

            order_result = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'executed_price': executed_price,
                'commission': executed_price * quantity * self.commission_rate,
                'timestamp': datetime.now(),
                'status': 'FILLED'
            }

            # Оновлення балансу
            self._update_balance(order_result)

            # Збереження в історії
            self.order_history.append(order_result)

            logger.info(f"✅ Ордер виконано: {side} {quantity} {symbol} at {executed_price}")
            return order_result

        except Exception as e:
            logger.error(f"❌ Помилка розміщення ордера {symbol}: {e}")
            return None

    def _simulate_execution(self, symbol: str, side: str, quantity: float) -> float:
        """Симуляція виконання ордера з slippage"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return 0

        # Додавання slippage
        if side == 'BUY':
            executed_price = current_price * (1 + self.slippage)
        else:  # SELL
            executed_price = current_price * (1 - self.slippage)

        return executed_price

    def _check_balance(self, symbol: str, side: str, quantity: float) -> bool:
        """Перевірка достатності балансу"""
        symbol_info = self.get_symbol_info(symbol)
        base_asset = symbol_info['base_asset']
        quote_asset = symbol_info['quote_asset']

        if side == 'BUY':
            # Потрібен quote asset (USDT)
            required = quantity * self.get_current_price(symbol) * (1 + self.commission_rate)
            available = self.balance.get(quote_asset, {}).get('free', 0)
            return available >= required
        else:  # SELL
            # Потрібен base asset (BTC)
            available = self.balance.get(base_asset, {}).get('free', 0)
            return available >= quantity

    def _update_balance(self, order_result: Dict):
        """Оновлення балансу після виконання ордера"""
        symbol_info = self.get_symbol_info(order_result['symbol'])
        base_asset = symbol_info['base_asset']
        quote_asset = symbol_info['quote_asset']

        quantity = order_result['quantity']
        executed_price = order_result['executed_price']
        commission = order_result['commission']

        if order_result['side'] == 'BUY':
            # Купівля: списуємо quote, додаємо base
            quote_cost = executed_price * quantity + commission

            self.balance[quote_asset]['free'] -= quote_cost
            self.balance[base_asset] = self.balance.get(base_asset, {'free': 0, 'locked': 0})
            self.balance[base_asset]['free'] += quantity

        else:  # SELL
            # Продаж: списуємо base, додаємо quote
            quote_received = executed_price * quantity - commission

            self.balance[base_asset]['free'] -= quantity
            self.balance[quote_asset]['free'] += quote_received

    def _round_to_step(self, value: float, step: float) -> float:
        """Округлення до step size"""
        return float(Decimal(str(value)).quantize(Decimal(str(step)), rounding=ROUND_DOWN))

    def start_websocket_stream(self, symbols: List[str], callback):
        """Запуск WebSocket стріму для real-time даних"""
        if not BINANCE_AVAILABLE:
            logger.error("python-binance не встановлено")
            return

        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket не доступний - робота без real-time стрімів")
            return

        try:
            self.socket_manager = BinanceSocketManager(self.client)

            # Стрім для кожного символу
            for symbol in symbols:
                stream_name = f"{symbol.lower()}@ticker"
                self.streams[symbol] = self.socket_manager.start_symbol_ticker_socket(
                    symbol, callback
                )

            # Запуск всіх стрімів
            self.socket_manager.start()

            logger.info(f"📡 WebSocket стріми запущено для: {symbols}")

        except Exception as e:
            logger.error(f"❌ Помилка запуску WebSocket: {e}")

    def stop_websocket_stream(self):
        """Зупинка WebSocket стрімів"""
        if self.socket_manager:
            self.socket_manager.stop()
            logger.info("📡 WebSocket стріми зупинено")

    def get_account_balance(self) -> Dict[str, Dict]:
        """Отримання поточного балансу рахунку"""
        return self.balance.copy()

    def get_positions_value(self) -> Dict[str, float]:
        """Отримання вартості позицій"""
        positions_value = {}

        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                positions_value[symbol] = position.quantity * current_price

        return positions_value

    def get_pnl_summary(self) -> Dict[str, Any]:
        """Зведення прибутків та збитків"""
        total_pnl = 0
        total_commission = 0
        total_trades = len(self.order_history)

        for order in self.order_history:
            if order['side'] == 'SELL':
                # Розрахунок P&L для закритих позицій
                # Спрощена версія - потрібно покращити
                pass

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'net_pnl': total_pnl - total_commission,
            'win_rate': 0.0  # Потрібно розрахувати
        }


class LiveStrategyTester:
    """
    Тестер стратегій в режимі реального часу на Binance Testnet
    """

    def __init__(self, trader: BinanceTestnetTrader, strategies: List[TradingStrategy],
                 risk_manager: Optional[RiskManager] = None):
        self.trader = trader
        self.strategies = strategies
        self.risk_manager = risk_manager or RiskManager()

        # Стан тестування
        self.is_running = False
        self.market_data = {}  # Кеш ринкових даних
        self.last_update = {}

        # Статистика
        self.total_signals = 0
        self.executed_signals = 0

    async def start_live_testing(self, symbols: List[str], update_interval: int = 60):
        """
        Запуск тестування стратегій в режимі реального часу

        Args:
            symbols: Список символів для тестування
            update_interval: Інтервал оновлення в секундах
        """
        self.is_running = True
        logger.info(f"🚀 Початок live тестування стратегій для: {symbols}")

        try:
            while self.is_running:
                # Оновлення ринкових даних
                await self._update_market_data(symbols)

                # Генерація сигналів стратегіями
                all_signals = await self._generate_strategy_signals(symbols)

                # Виконання сигналів
                await self._execute_signals(all_signals)

                # Очікування наступного оновлення
                await asyncio.sleep(update_interval)

        except Exception as e:
            logger.error(f"❌ Помилка в live тестуванні: {e}")
        finally:
            self.trader.stop_websocket_stream()

    async def _update_market_data(self, symbols: List[str]):
        """Оновлення ринкових даних"""
        for symbol in symbols:
            try:
                # Отримання свіжих даних
                current_price = self.trader.get_current_price(symbol)
                recent_trades = self.trader.get_recent_trades(symbol, limit=50)

                if current_price:
                    self.market_data[symbol] = {
                        'price': current_price,
                        'trades': recent_trades,
                        'timestamp': datetime.now()
                    }

            except Exception as e:
                logger.error(f"❌ Помилка оновлення даних {symbol}: {e}")

    async def _generate_strategy_signals(self, symbols: List[str]) -> Dict[str, List[TradeSignal]]:
        """Генерація сигналів від всіх стратегій"""
        all_signals = {}

        for strategy in self.strategies:
            strategy_symbols = [s for s in symbols if s in strategy.symbols]

            if not strategy_symbols:
                continue

            try:
                # Підготовка даних для стратегії
                market_snapshot = {}
                predictions = {}

                for symbol in strategy_symbols:
                    if symbol in self.market_data:
                        # Спрощена версія - потрібні повні OHLCV дані
                        market_snapshot[symbol] = pd.DataFrame()  # TODO: Додати реальні дані
                        predictions[symbol] = {
                            'change_percent': 0.01,  # Спрощений прогноз
                            'confidence': 0.7
                        }

                # Генерація сигналів
                signals = strategy.analyze_market(market_snapshot, predictions)
                all_signals[strategy.name] = list(signals.values())

                self.total_signals += len(signals)

            except Exception as e:
                logger.error(f"❌ Помилка генерації сигналів {strategy.name}: {e}")

        return all_signals

    async def _execute_signals(self, all_signals: Dict[str, List[TradeSignal]]):
        """Виконання торгових сигналів"""
        for strategy_name, signals in all_signals.items():
            for signal in signals:
                try:
                    # Перевірка ризику
                    can_trade, reason = self.risk_manager.can_open_position(
                        capital=10000,  # Спрощена версія
                        position_size=signal.quantity * signal.entry_price if signal.quantity else 100,
                        current_positions=len(self.trader.positions)
                    )

                    if not can_trade:
                        logger.info(f"⚠️ Сигнал {strategy_name} відхилено: {reason}")
                        continue

                    # Виконання ордера
                    if signal.action == TradeAction.BUY:
                        order = self.trader.place_market_order(
                            signal.symbol, 'BUY', signal.quantity, test=True
                        )
                    elif signal.action == TradeAction.SELL:
                        order = self.trader.place_market_order(
                            signal.symbol, 'SELL', signal.quantity, test=True
                        )

                    if order:
                        self.executed_signals += 1
                        logger.info(f"✅ Виконано сигнал {strategy_name}: {signal.action.value} {signal.symbol}")

                except Exception as e:
                    logger.error(f"❌ Помилка виконання сигналу {strategy_name}: {e}")

    def stop_testing(self):
        """Зупинка тестування"""
        self.is_running = False
        logger.info("🛑 Live тестування зупинено")

    def get_testing_stats(self) -> Dict[str, Any]:
        """Статистика тестування"""
        return {
            'total_signals': self.total_signals,
            'executed_signals': self.executed_signals,
            'execution_rate': self.executed_signals / max(self.total_signals, 1),
            'account_balance': self.trader.get_account_balance(),
            'pnl_summary': self.trader.get_pnl_summary(),
            'active_positions': len(self.trader.positions)
        }


# Приклад використання
async def example_live_testing():
    """Приклад запуску live тестування"""

    # Ініціалізація трейдера
    trader = BinanceTestnetTrader(
        api_key="your_test_api_key",
        api_secret="your_test_api_secret",
        test_mode=True
    )

    # Створення стратегій
    from strategies.scalping import ScalpingStrategy
    from strategies.day_trading import DayTradingStrategy

    strategies = [
        ScalpingStrategy(['BTCUSDT']),
        DayTradingStrategy(['BTCUSDT', 'ETHUSDT'])
    ]

    # Ініціалізація тестера
    tester = LiveStrategyTester(trader, strategies)

    # Запуск тестування
    try:
        await tester.start_live_testing(['BTCUSDT', 'ETHUSDT'], update_interval=30)
    except KeyboardInterrupt:
        tester.stop_testing()

    # Вивід результатів
    stats = tester.get_testing_stats()
    print("📊 Результати тестування:", stats)


if __name__ == "__main__":
    # Запуск прикладу
    asyncio.run(example_live_testing())