#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простий trading скрипт для Random Forest моделей
Використовує тільки BTCUSDT (81.15% accuracy)
"""
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from binance.client import Client

# Додаємо шлях до проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.rust_features import RustFeatureEngineer
from training.simple_trend_classifier import SimpleTrendClassifier
from telegram_bot import telegram_notifier
from websocket_manager import BinanceFuturesWebSocket
from mtf_analyzer import MultiTimeframeAnalyzer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database
try:
    from optimized.database.connection import DatabaseConnection, save_position, save_trade
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️ Database не доступна")


class SimpleTradingBot:
    """Простий торговий бот з Random Forest прогнозами"""

    def __init__(self, symbols: list = None, testnet: bool = True, enable_trading: bool = False):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']  # Обмежено до 6 символів
        self.testnet = testnet
        self.enable_trading = enable_trading  # Чи виконувати реальні угоди

        # API ключі з .env
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('FUTURES_API_KEY')
        api_secret = os.getenv('FUTURES_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("❌ Потрібні FUTURES_API_KEY та FUTURES_API_SECRET в .env")

        # Binance клієнт
        self.client = Client(api_key, api_secret, testnet=testnet)
        logger.info(f"✅ Binance client ({'TESTNET' if testnet else 'PRODUCTION'})")

        # Моделі для кожного символу
        self.models = {}
        self.scalers = {}
        self.feature_names_dict = {}
        self.feature_engineer = RustFeatureEngineer()

        # Multi-timeframe аналіз
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.use_mtf = True  # Увімкнути MTF за замовчуванням

        # WebSocket для real-time оновлень
        self.websocket = BinanceFuturesWebSocket(self.client, testnet=testnet)
        self.websocket.on_order_update = self._on_order_update
        self.websocket.on_account_update = self._on_account_update

        # Позиції та статистика
        self.positions = {}  # {symbol: position_data}
        self.balance = 0.0
        self.trades = []

        # 🛡️ ЗАХИСТ ВІД OVERTRADING
        self.closed_positions = {}  # {symbol: {'time': datetime, 'reason': 'SL/TP', 'pnl': float}}
        self.symbol_blacklist = {}  # {symbol: datetime} - коли можна знову торгувати
        self.cooldown_after_sl = 3600  # 1 година після SL (секунди)
        self.cooldown_after_tp = 1800  # 30 хв після TP (секунди)
        self.max_daily_losses_per_symbol = 3  # Максимум 3 програшні угоди на день

        # Налаштування trading
        self.min_confidence = 0.80  # Мінімальна впевненість для відкриття (80%)
        self.position_size_usd = 50.0  # Зменшено для тестування $50
        self.stop_loss_pct = 0.02   # 2% stop-loss
        self.take_profit_pct = 0.05 # 5% take-profit
        self.leverage = 25          # Плече 25x

        if enable_trading:
            logger.warning("⚠️ РЕАЛЬНІ УГОДИ УВІМКНЕНІ!")
            # Встановлюємо плече для всіх символів
            for symbol in self.symbols:
                try:
                    self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
                    logger.info(f"⚡ {symbol}: плече встановлено {self.leverage}x")
                except Exception as e:
                    logger.warning(f"⚠️ Не вдалося встановити плече для {symbol}: {e}")
        else:
            logger.info("ℹ️ Demo режим (угоди не виконуються)")

    def load_models(self):
        """Завантаження моделей для всіх символів"""
        for symbol in self.symbols:
            try:
                model_dir = Path(f'models/simple_trend_{symbol}')
                if not model_dir.exists():
                    logger.warning(f"⚠️ Модель для {symbol} не знайдено, пропускаємо")
                    continue

                # Знайти файли моделі
                pkl_files = list(model_dir.glob('model_*.pkl'))
                if not pkl_files:
                    logger.warning(f"⚠️ Файли моделі для {symbol} не знайдено")
                    continue

                model_path = str(pkl_files[0])
                timeframe = pkl_files[0].stem.split('_')[-1]
                scaler_path = str(model_dir / f'scaler_{symbol}_{timeframe}.pkl')
                features_path = str(model_dir / f'features_{symbol}_{timeframe}.pkl')

                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.feature_names_dict[symbol] = joblib.load(features_path)

                logger.info(f"✅ {symbol}: модель завантажено ({len(self.feature_names_dict[symbol])} features)")

            except Exception as e:
                logger.error(f"❌ Помилка завантаження моделі {symbol}: {e}")

        if not self.models:
            raise RuntimeError("❌ Жодна модель не завантажена!")

        logger.info(f"✅ Завантажено моделей: {len(self.models)}/{len(self.symbols)}")

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Округлення quantity згідно з правилами Binance"""
        # Типові precision для різних символів
        precision_map = {
            'BTCUSDT': 3,
            'ETHUSDT': 3,
            'BNBUSDT': 2,
            'SOLUSDT': 2,  # FIX: було 1, треба 2
            'ADAUSDT': 0,
            'DOGEUSDT': 0,
            'XRPUSDT': 1,
            'LTCUSDT': 3,
            'LINKUSDT': 2,
            'MATICUSDT': 0,
            'DOTUSDT': 1,
            'UNIUSDT': 2,
            'ATOMUSDT': 2,
            'ETCUSDT': 2,
            'XLMUSDT': 0,
            'ALGOUSDT': 0,
            'VETUSDT': 0,
            'FILUSDT': 2,
            'TRXUSDT': 0,
            'AVAXUSDT': 2,
        }

        precision = precision_map.get(symbol, 2)  # Default 2
        rounded = round(quantity, precision)

        # Для precision=0 повертаємо int (БЕЗ decimal point)
        if precision == 0:
            return int(rounded)
        return rounded

    def format_quantity_for_binance(self, symbol: str, quantity: float) -> str:
        """Форматування quantity для Binance API з правильною precision"""
        precision_map = {
            'BTCUSDT': 3, 'ETHUSDT': 3, 'BNBUSDT': 2, 'SOLUSDT': 2,  # FIX
            'ADAUSDT': 0, 'DOGEUSDT': 0, 'XRPUSDT': 1, 'LTCUSDT': 3,
            'LINKUSDT': 2, 'MATICUSDT': 0, 'DOTUSDT': 1, 'UNIUSDT': 2,
            'ATOMUSDT': 2, 'ETCUSDT': 2, 'XLMUSDT': 0, 'ALGOUSDT': 0,
            'VETUSDT': 0, 'FILUSDT': 2, 'TRXUSDT': 0, 'AVAXUSDT': 2
        }
        precision = precision_map.get(symbol, 2)

        if precision == 0:
            return str(int(quantity))
        else:
            return f"{quantity:.{precision}f}"

    async def get_balance(self) -> float:
        """Отримання балансу"""
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_account()
            )
            balance = float(account['totalWalletBalance'])
            logger.info(f"💰 Баланс: ${balance:.2f} USDT")
            return balance
        except Exception as e:
            logger.error(f"❌ Помилка отримання балансу: {e}")
            return 0.0

    async def _on_order_update(self, order_info: dict):
        """Колбек для WebSocket order updates"""
        try:
            symbol = order_info['symbol']
            status = order_info['status']
            side = order_info['side']
            filled_qty = order_info['filled_quantity']
            avg_price = order_info['avg_price']

            # Логування виконаних ордерів
            if status == 'FILLED' and avg_price:
                logger.info(
                    f"🔔 WS: Ордер виконано - {symbol} {side} "
                    f"{filled_qty} @ ${avg_price:.2f}"
                )

                # Відправка в Telegram
                await telegram_notifier.send_message(
                    f"🔔 ORDER FILLED (WebSocket)\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Quantity: {filled_qty}\n"
                    f"Price: ${avg_price:.2f}\n"
                    f"Order ID: {order_info['order_id']}"
                )

            # SL/TP виконання
            order_type = order_info.get('order_type', '')
            if status == 'FILLED' and order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                # Отримуємо PnL позиції перед закриттям
                position = await self.check_position(symbol)
                pnl = position['unrealized_pnl'] if position else 0.0

                if order_type == 'STOP_MARKET':
                    logger.warning(f"🛑 WS: Stop-Loss спрацював - {symbol} (PnL: ${pnl:+.2f})")
                    await telegram_notifier.send_message(
                        f"🛑 STOP-LOSS HIT\n"
                        f"Symbol: {symbol}\n"
                        f"Price: ${avg_price:.2f}\n"
                        f"PnL: ${pnl:+.2f}"
                    )
                    # 🛡️ Додаємо в blacklist після SL
                    self.add_to_blacklist(symbol, 'SL', pnl)

                else:
                    logger.info(f"🎯 WS: Take-Profit спрацював - {symbol} (PnL: ${pnl:+.2f})")
                    await telegram_notifier.send_message(
                        f"🎯 TAKE-PROFIT HIT\n"
                        f"Symbol: {symbol}\n"
                        f"Price: ${avg_price:.2f}\n"
                        f"PnL: ${pnl:+.2f}"
                    )
                    # 🛡️ Додаємо в blacklist після TP (коротший cooldown)
                    self.add_to_blacklist(symbol, 'TP', pnl)

        except Exception as e:
            logger.error(f"❌ Помилка обробки order update: {e}")

    async def _on_account_update(self, account_data: dict):
        """Колбек для WebSocket account updates"""
        try:
            # Оновлення балансу з WebSocket
            balances = account_data.get('B', [])
            for balance in balances:
                if balance['a'] == 'USDT':
                    new_balance = float(balance['wb'])
                    if abs(new_balance - self.balance) > 0.01:
                        logger.info(f"💰 WS: Balance оновлено ${new_balance:.2f}")
                        self.balance = new_balance
        except Exception as e:
            logger.error(f"❌ Помилка обробки account update: {e}")

    async def get_market_data(self, symbol: str, interval: str = '4h', limit: int = 500) -> pd.DataFrame:
        """Завантаження ринкових даних"""
        try:
            klines = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Конвертація типів
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"📊 Завантажено {len(df)} свічок")
            return df

        except Exception as e:
            logger.error(f"❌ Помилка завантаження даних: {e}")
            return pd.DataFrame()

    async def get_atr(self, symbol: str, period: int = 14) -> float:
        """Отримання ATR для динамічних SL/TP"""
        try:
            df = await self.get_market_data(symbol, '1h', 100)  # 1h для ATR
            if df.empty:
                return 0.01  # Default fallback

            # Розрахунок ATR через Rust
            df_features = self.feature_engineer.calculate_all(
                df, atr_periods=[period]
            )

            atr_col = f'atr_{period}'
            if atr_col in df_features.columns:
                current_atr = df_features[atr_col].iloc[-1]
                if pd.isna(current_atr):
                    return 0.01
                return current_atr
            else:
                logger.warning(f"⚠️ ATR не розраховано для {symbol}")
                return 0.01

        except Exception as e:
            logger.error(f"❌ Помилка розрахунку ATR для {symbol}: {e}")
            return 0.01

    async def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """Прогноз напрямку руху"""
        try:
            if symbol not in self.models:
                logger.warning(f"⚠️ Модель для {symbol} не завантажена")
                return None

            # Розрахунок features через Rust (точно як у SimpleTrendClassifier)
            df_features = self.feature_engineer.calculate_all(
                df,
                sma_periods=[5, 10, 20, 50, 100, 200],
                ema_periods=[9, 12, 21, 26, 50],
                rsi_periods=[7, 14, 21, 28],
                atr_periods=[14, 21],
            )

            # Price relative to MAs
            for period in [5, 10, 20, 50, 100, 200]:
                ma_col = f'sma_{period}'
                if ma_col in df_features.columns:
                    df_features[f'price_vs_sma{period}'] = (df['close'] / df_features[ma_col] - 1) * 100

            # MA crossovers
            if 'sma_50' in df_features.columns and 'sma_200' in df_features.columns:
                df_features['golden_cross'] = (df_features['sma_50'] > df_features['sma_200']).astype(int)

            if 'ema_12' in df_features.columns and 'ema_26' in df_features.columns:
                df_features['macd_cross'] = (df_features['ema_12'] > df_features['ema_26']).astype(int)

            # Volume features
            if 'volume' in df.columns:
                df_features['volume_sma20'] = df['volume'].rolling(20).mean()
                df_features['volume_trend'] = df['volume'] / df_features['volume_sma20']

            # RSI levels
            if 'rsi_14' in df_features.columns:
                df_features['rsi_overbought'] = (df_features['rsi_14'] > 70).astype(int)
                df_features['rsi_oversold'] = (df_features['rsi_14'] < 30).astype(int)

            # Momentum
            df_features['momentum_5'] = df['close'].pct_change(5) * 100
            df_features['momentum_10'] = df['close'].pct_change(10) * 100
            df_features['momentum_20'] = df['close'].pct_change(20) * 100

            # Volatility
            df_features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100

            # Очистка NaN
            df_features = df_features.dropna()

            if len(df_features) < 10:
                logger.warning("⚠️ Недостатньо даних для прогнозу")
                return None

            # Використовуємо тільки ті features, які були при тренуванні
            feature_names = self.feature_names_dict[symbol]
            missing_features = [f for f in feature_names if f not in df_features.columns]
            if missing_features:
                logger.warning(f"⚠️ Відсутні features для {symbol}: {missing_features[:5]}...")
                return None

            # Беремо останній рядок з потрібними features
            X = df_features[feature_names].iloc[-1:].values

            # Скалювання
            X_scaled = self.scalers[symbol].transform(X)

            # Прогноз
            prediction = self.models[symbol].predict(X_scaled)[0]
            proba = self.models[symbol].predict_proba(X_scaled)[0]

            current_price = df['close'].iloc[-1]

            result = {
                'symbol': symbol,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': max(proba),
                'proba_down': proba[0],
                'proba_up': proba[1] if len(proba) > 1 else 0,
                'current_price': current_price,
                'timestamp': datetime.now()
            }

            logger.info(
                f"🤖 {symbol}: {result['prediction']} "
                f"(confidence: {result['confidence']:.2%}, "
                f"price: ${current_price:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Помилка прогнозу {symbol}: {e}", exc_info=True)
            return None

    async def predict_mtf(self, symbol: str) -> dict:
        """
        Multi-timeframe прогноз
        Аналізує 4h, 1h, 15m і комбінує сигнали
        """
        try:
            if not self.use_mtf:
                # Якщо MTF вимкнено, використовуємо стандартний прогноз
                df = await self.get_market_data(symbol, interval='4h')
                return await self.predict(symbol, df)

            # Завантаження даних для всіх таймфреймів
            timeframes = ['4h', '1h', '15m']
            predictions = {}

            for tf in timeframes:
                df = await self.get_market_data(symbol, interval=tf, limit=1000)

                if df.empty:
                    logger.warning(f"⚠️ {symbol} {tf}: немає даних")
                    continue

                pred = await self.predict(symbol, df)
                if pred:
                    predictions[tf] = pred

            # Перевірка наявності всіх прогнозів
            if len(predictions) < 3:
                logger.warning(f"⚠️ {symbol}: недостатньо MTF даних ({len(predictions)}/3)")
                # Fallback на 4h якщо є
                return predictions.get('4h')

            # Комбінування через MTF аналізатор
            mtf_result = self.mtf_analyzer.analyze(predictions, require_alignment=True)

            if not mtf_result:
                logger.info(f"⚠️ {symbol}: MTF не дав чіткого сигналу")
                return None

            return mtf_result

        except Exception as e:
            logger.error(f"❌ Помилка MTF прогнозу {symbol}: {e}", exc_info=True)
            return None

    async def check_position(self, symbol: str) -> dict:
        """Перевірка поточної позиції"""
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_position_information(symbol=symbol)
            )

            for pos in positions:
                if pos['symbol'] == symbol:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0.0001:  # Позиція відкрита
                        return {
                            'symbol': symbol,
                            'amount': amt,
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'side': 'LONG' if amt > 0 else 'SHORT'
                        }

            return None

        except Exception as e:
            logger.error(f"❌ Помилка перевірки позиції {symbol}: {e}")
            return None

    def is_symbol_blacklisted(self, symbol: str) -> bool:
        """Перевірка чи символ в blacklist"""
        if symbol not in self.symbol_blacklist:
            return False

        # Перевірка чи минув cooldown період
        blacklist_until = self.symbol_blacklist[symbol]
        now = datetime.now()

        if now >= blacklist_until:
            # Cooldown минув, видаляємо з blacklist
            del self.symbol_blacklist[symbol]
            logger.info(f"✅ {symbol}: cooldown період завершено")
            return False

        # Ще в blacklist
        remaining = (blacklist_until - now).total_seconds() / 60
        logger.info(f"⏸️ {symbol}: в cooldown ще {remaining:.1f} хв")
        return True

    def add_to_blacklist(self, symbol: str, reason: str, pnl: float):
        """Додати символ в blacklist після закриття позиції"""
        from datetime import datetime, timedelta

        # Визначаємо тривалість cooldown
        if reason == 'SL' or pnl < 0:
            cooldown_seconds = self.cooldown_after_sl  # 1 година після SL
            cooldown_label = "1 година"
        else:
            cooldown_seconds = self.cooldown_after_tp  # 30 хв після TP
            cooldown_label = "30 хв"

        blacklist_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        self.symbol_blacklist[symbol] = blacklist_until

        # Зберігаємо історію закритої позиції
        if symbol not in self.closed_positions:
            self.closed_positions[symbol] = []

        self.closed_positions[symbol].append({
            'time': datetime.now(),
            'reason': reason,
            'pnl': pnl
        })

        logger.warning(f"🚫 {symbol}: додано в blacklist на {cooldown_label} (reason: {reason}, PnL: ${pnl:+.2f})")

    def get_daily_losses_count(self, symbol: str) -> int:
        """Підрахунок програшних угод за сьогодні"""
        if symbol not in self.closed_positions:
            return 0

        from datetime import datetime, timedelta
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        losses_today = [
            pos for pos in self.closed_positions[symbol]
            if pos['time'] >= today_start and pos['pnl'] < 0
        ]

        return len(losses_today)

    async def open_long_position(self, symbol: str, price: float, confidence: float):
        """Відкриття LONG позиції"""
        try:
            # 🛡️ Перевірка blacklist
            if self.is_symbol_blacklisted(symbol):
                logger.warning(f"⏸️ {symbol}: в cooldown періоді, пропускаємо")
                return False

            # 🛡️ Перевірка денного ліміту програшів
            daily_losses = self.get_daily_losses_count(symbol)
            if daily_losses >= self.max_daily_losses_per_symbol:
                logger.warning(f"🚫 {symbol}: досягнуто ліміт програшів за день ({daily_losses}/{self.max_daily_losses_per_symbol})")
                return False

            # Розрахунок розміру позиції: $500 позиція з leverage 25x
            # Margin = $500 / 25 = $20
            position_value = self.position_size_usd  # $500
            margin_required = position_value / self.leverage  # $20
            quantity = position_value / price
            quantity = self.round_quantity(symbol, quantity)  # Округлення

            if quantity * price < 10:  # Мінімальна сума угоди $10
                logger.warning(f"⚠️ Занадто маленька сума: ${quantity * price:.2f}")
                return False

            logger.info(f"📈 Відкриваємо LONG {symbol}: {quantity} @ ${price:.2f}")
            logger.info(f"   Margin: ${margin_required:.2f} | Position: ${position_value:.2f} ({self.leverage}x)")

            if not self.enable_trading:
                logger.info("ℹ️ Demo режим - угода НЕ виконана")
                await telegram_notifier.send_message(
                    f"📈 DEMO BUY Signal\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"Quantity: {quantity:.6f}\n"
                    f"Confidence: {confidence:.2%}"
                )
                return False

            # Виконання ордера
            quantity_str = self.format_quantity_for_binance(symbol, quantity)

            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity_str
                )
            )

            logger.info(f"✅ Ордер виконано: {order['orderId']}")

            # Отримуємо реальну ціну з позиції (треба почекати оновлення)
            await asyncio.sleep(1.5)  # Збільшена затримка для Testnet

            position_info = await self.check_position(symbol)
            if position_info and position_info['entry_price'] > 0:
                filled_price = position_info['entry_price']
                filled_qty = abs(position_info['amount'])
                logger.info(f"💰 Отримано з позиції: {filled_qty:.6f} @ ${filled_price:.2f}")
            else:
                # Fallback: використовуємо розрахункові значення
                filled_price = price
                filled_qty = quantity
                logger.warning(f"⚠️ Позиція ще не оновилась, використовую розрахункові: {filled_qty:.6f} @ ${filled_price:.2f}")

            # Stop-loss і Take-profit ціни (динамічні на основі ATR)
            atr = await self.get_atr(symbol, 14)
            sl_distance = 2.0 * atr  # 2 ATR для SL
            tp_distance = 4.0 * atr  # 4 ATR для TP

            sl_price = filled_price - sl_distance  # Для LONG: нижче entry
            tp_price = filled_price + tp_distance  # Для LONG: вище entry

            # Виставляємо STOP_MARKET для SL
            try:
                sl_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='SELL',
                        type='STOP_MARKET',
                        stopPrice=round(sl_price, 2),
                        quantity=self.format_quantity_for_binance(symbol, filled_qty),
                        closePosition=False
                    )
                )
                logger.info(f"🛑 Stop-loss виставлено: ${sl_price:.2f} (orderId: {sl_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Помилка виставлення SL: {e}")

            # Виставляємо TAKE_PROFIT_MARKET для TP
            try:
                tp_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='SELL',
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=round(tp_price, 2),
                        quantity=self.format_quantity_for_binance(symbol, filled_qty),
                        closePosition=False
                    )
                )
                logger.info(f"🎯 Take-profit виставлено: ${tp_price:.2f} (orderId: {tp_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Помилка виставлення TP: {e}")

            logger.info(f"🛑 Stop-loss: ${sl_price:.2f}")
            logger.info(f"🎯 Take-profit: ${tp_price:.2f}")

            # Розрахунок для Telegram
            position_value = filled_qty * filled_price
            margin_used = position_value / self.leverage

            # Telegram notification
            await telegram_notifier.send_message(
                f"✅ OPENED LONG\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${filled_price:.2f}\n"
                f"Quantity: {filled_qty:.6f}\n"
                f"Position: ${position_value:.2f}\n"
                f"Margin: ${margin_used:.2f} ({self.leverage}x)\n"
                f"SL: ${sl_price:.2f} (-{self.stop_loss_pct*100:.1f}%)\n"
                f"TP: ${tp_price:.2f} (+{self.take_profit_pct*100:.1f}%)\n"
                f"Confidence: {confidence:.2%}"
            )

            # Збереження в БД
            if DB_AVAILABLE:
                try:
                    db = DatabaseConnection()
                    await save_position(db, {
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_price': filled_price,
                        'quantity': filled_qty,
                        'stop_loss': sl_price,
                        'take_profit': tp_price,
                        'status': 'open',
                        'strategy': 'ML_4h',
                        'entry_time': datetime.now(),
                        'signal_id': None,
                        'metadata': {
                            'confidence': float(confidence),  # numpy.float64 → float
                            'leverage': self.leverage
                        }
                    })
                except Exception as e:
                    logger.error(f"❌ Помилка збереження в БД: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Помилка відкриття позиції {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR opening {symbol}: {str(e)[:100]}")
            return False

    async def open_short_position(self, symbol: str, price: float, confidence: float):
        """Відкриття SHORT позиції (продаж на зниження)"""
        try:
            # 🛡️ Перевірка blacklist
            if self.is_symbol_blacklisted(symbol):
                logger.warning(f"⏸️ {symbol}: в cooldown періоді, пропускаємо")
                return False

            # 🛡️ Перевірка денного ліміту програшів
            daily_losses = self.get_daily_losses_count(symbol)
            if daily_losses >= self.max_daily_losses_per_symbol:
                logger.warning(f"🚫 {symbol}: досягнуто ліміт програшів за день ({daily_losses}/{self.max_daily_losses_per_symbol})")
                return False

            # Розрахунок розміру позиції: $500 позиція з leverage 25x
            position_value = self.position_size_usd  # $500
            margin_required = position_value / self.leverage  # $20
            quantity = position_value / price
            quantity = self.round_quantity(symbol, quantity)  # Округлення

            # Мінімальна угода $10
            if position_value < 10:
                logger.warning(f"⚠️ {symbol}: розмір позиції ${position_value:.2f} < $10, пропускаємо")
                return False

            logger.info(f"📉 Відкриваємо SHORT {symbol}: {quantity:.6f} @ ${price:.2f}")
            logger.info(f"   Margin: ${margin_required:.2f} | Position: ${position_value:.2f} ({self.leverage}x)")
            logger.info(f"   Confidence: {confidence:.2%}")

            if not self.enable_trading:
                logger.info("ℹ️ Demo режим - угода НЕ виконана")
                await telegram_notifier.send_message(
                    f"📉 DEMO SHORT Signal\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"Quantity: {quantity:.6f}\n"
                    f"Margin: ${margin_required:.2f}\n"
                    f"Position: ${position_value:.2f} ({self.leverage}x)\n"
                    f"Confidence: {confidence:.2%}"
                )
                return False

            # Виконання SELL ордера
            quantity_str = self.format_quantity_for_binance(symbol, quantity)

            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity_str
                )
            )

            logger.info(f"✅ SHORT відкрито: {order['orderId']}")

            # Отримуємо реальну ціну з позиції (з більшою затримкою для Testnet)
            await asyncio.sleep(1.5)  # Збільшена затримка

            position_info = await self.check_position(symbol)
            if position_info and position_info['entry_price'] > 0:
                filled_price = position_info['entry_price']
                filled_qty = abs(position_info['amount'])
                logger.info(f"💰 Отримано з позиції: {filled_qty:.6f} @ ${filled_price:.2f}")
            else:
                filled_price = price
                filled_qty = quantity
                logger.warning(f"⚠️ Позиція ще не оновилась, використовую розрахункові значення")

            # SL/TP для SHORT (динамічні на основі ATR)
            atr = await self.get_atr(symbol, 14)
            sl_distance = 2.0 * atr  # 2 ATR для SL
            tp_distance = 4.0 * atr  # 4 ATR для TP

            sl_price = filled_price + sl_distance  # Для SHORT: вище entry
            tp_price = filled_price - tp_distance  # Для SHORT: нижче entry

            # Виставляємо STOP_MARKET для SL (BUY to close SHORT)
            try:
                sl_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='BUY',  # BUY для закриття SHORT
                        type='STOP_MARKET',
                        stopPrice=round(sl_price, 2),
                        quantity=self.format_quantity_for_binance(symbol, filled_qty),
                        closePosition=False
                    )
                )
                logger.info(f"🛑 Stop-loss виставлено: ${sl_price:.2f} (orderId: {sl_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Помилка виставлення SL: {e}")

            # Виставляємо TAKE_PROFIT_MARKET для TP (BUY to close SHORT)
            try:
                tp_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='BUY',  # BUY для закриття SHORT
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=round(tp_price, 2),
                        quantity=self.format_quantity_for_binance(symbol, filled_qty),
                        closePosition=False
                    )
                )
                logger.info(f"🎯 Take-profit виставлено: ${tp_price:.2f} (orderId: {tp_order['orderId']})")
            except Exception as e:
                logger.error(f"❌ Помилка виставлення TP: {e}")

            # Розрахунок для Telegram
            position_value = filled_qty * filled_price
            margin_used = position_value / self.leverage

            # Telegram notification
            await telegram_notifier.send_message(
                f"📉 SHORT OPENED\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${filled_price:.2f}\n"
                f"Quantity: {filled_qty:.6f}\n"
                f"Position: ${position_value:.2f}\n"
                f"Margin: ${margin_used:.2f} ({self.leverage}x)\n"
                f"SL: ${sl_price:.2f} (+{self.stop_loss_pct*100:.1f}%)\n"
                f"TP: ${tp_price:.2f} (-{self.take_profit_pct*100:.1f}%)\n"
                f"Confidence: {confidence:.2%}"
            )

            # Збереження в БД
            if DB_AVAILABLE:
                try:
                    db = DatabaseConnection()
                    await save_position(db, {
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_price': filled_price,
                        'quantity': filled_qty,
                        'stop_loss': sl_price,
                        'take_profit': tp_price,
                        'status': 'open',
                        'strategy': 'ML_4h',
                        'entry_time': datetime.now(),
                        'signal_id': None,
                        'metadata': {
                            'confidence': float(confidence),  # numpy.float64 → float
                            'leverage': self.leverage
                        }
                    })
                except Exception as e:
                    logger.error(f"❌ Помилка збереження в БД: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Помилка відкриття SHORT {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR opening SHORT {symbol}: {str(e)[:100]}")
            return False

    async def close_long_position(self, symbol: str, position: dict, price: float, reason: str):
        """Закриття LONG позиції"""
        try:
            quantity = abs(position['amount'])
            entry_price = position['entry_price']
            pnl = position['unrealized_pnl']

            logger.info(f"📉 Закриваємо LONG {symbol}: {quantity} @ ${price:.2f} (PnL: ${pnl:.2f})")

            if not self.enable_trading:
                logger.info("ℹ️ Demo режим - угода НЕ виконана")
                await telegram_notifier.send_message(
                    f"📉 DEMO SELL Signal\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"PnL: ${pnl:.2f}\n"
                    f"Reason: {reason}"
                )
                return False

            # Виконання ордера
            quantity_str = self.format_quantity_for_binance(symbol, quantity)

            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity_str
                )
            )

            logger.info(f"✅ Позицію закрито: {order['orderId']}")

            # Розрахунок P&L %
            pnl_pct = ((price - entry_price) / entry_price) * 100

            # Telegram notification
            emoji = "💰" if pnl > 0 else "📉"
            await telegram_notifier.send_message(
                f"{emoji} CLOSED LONG\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Exit: ${price:.2f}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                f"Reason: {reason}"
            )

            # Збереження в БД
            if DB_AVAILABLE:
                try:
                    db = DatabaseConnection()
                    await save_trade(db, {
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'quantity': quantity,
                        'entry_time': None,  # TODO: зберігати з position
                        'exit_time': datetime.now(),
                        'pnl': pnl,
                        'pnl_percentage': pnl_pct,
                        'strategy': 'ML_4h',
                        'exit_reason': reason,
                        'position_id': None,
                        'signal_id': None,
                        'fees': 0.0,
                        'metadata': {}
                    })
                except Exception as e:
                    logger.error(f"❌ Помилка збереження в БД: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Помилка закриття позиції {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR closing {symbol}: {str(e)[:100]}")
            return False

    async def close_short_position(self, symbol: str, position: dict, price: float, reason: str):
        """Закриття SHORT позиції"""
        try:
            quantity = abs(position['amount'])
            entry_price = position['entry_price']
            pnl = position['unrealized_pnl']

            logger.info(f"📈 Закриваємо SHORT {symbol}: {quantity} @ ${price:.2f} (PnL: ${pnl:.2f})")

            if not self.enable_trading:
                logger.info("ℹ️ Demo режим - угода НЕ виконана")
                await telegram_notifier.send_message(
                    f"📈 DEMO BUY to Close SHORT\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"PnL: ${pnl:.2f}\n"
                    f"Reason: {reason}"
                )
                return False

            # Виконання BUY ордера (закриваємо SHORT)
            quantity_str = self.format_quantity_for_binance(symbol, quantity)

            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity_str
                )
            )

            logger.info(f"✅ SHORT закрито: {order['orderId']}")

            # Розрахунок P&L %
            pnl_pct = ((entry_price - price) / entry_price) * 100  # Протилежно LONG

            # Telegram notification
            await telegram_notifier.send_message(
                f"📈 SHORT CLOSED\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Exit: ${price:.2f}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                f"Reason: {reason}"
            )

            # Збереження в БД
            if DB_AVAILABLE:
                try:
                    db = DatabaseConnection()
                    await save_trade(db, {
                        'symbol': symbol,
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'quantity': quantity,
                        'entry_time': None,
                        'exit_time': datetime.now(),
                        'pnl': pnl,
                        'pnl_percentage': pnl_pct,
                        'strategy': 'ML_4h',
                        'exit_reason': reason,
                        'position_id': None,
                        'signal_id': None,
                        'fees': 0.0,
                        'metadata': {}
                    })
                except Exception as e:
                    logger.error(f"❌ Помилка збереження в БД: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Помилка закриття SHORT {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR closing SHORT {symbol}: {str(e)[:100]}")
            return False

    async def run_single_check(self):
        """Одноразова перевірка для демо режиму"""
        try:
            logger.info("🔍 Перевірка поточного стану позицій...")

            total_positions = 0
            total_pnl = 0.0

            for symbol in self.symbols:
                try:
                    position = await self.check_position(symbol)
                    if position:
                        total_positions += 1
                        total_pnl += position['unrealized_pnl']

                        side_emoji = "📈" if position['side'] == 'LONG' else "📉"
                        pnl_emoji = "💰" if position['unrealized_pnl'] > 0 else "📉"

                        logger.info(
                            f"{side_emoji} {symbol}: {position['side']} "
                            f"{abs(position['amount']):.4f} @ ${position['entry_price']:.2f} "
                            f"PnL: {pnl_emoji} ${position['unrealized_pnl']:+.2f}"
                        )
                    else:
                        logger.info(f"⚪ {symbol}: немає позиції")

                except Exception as e:
                    logger.error(f"❌ Помилка перевірки {symbol}: {e}")

            if total_positions > 0:
                logger.info(f"\n📊 Підсумок: {total_positions} позицій, загальний PnL: ${total_pnl:+.2f}")
            else:
                logger.info("\n✅ Немає відкритих позицій")

        except Exception as e:
            logger.error(f"❌ Помилка run_single_check: {e}")

    async def run(self, interval_seconds: int = 3600):
        """Головний цикл"""
        logger.info("=" * 60)
        logger.info(f"🚀 ЗАПУСК TRADING BOT")
        logger.info(f"Символи: {', '.join(self.symbols)}")
        logger.info(f"MTF аналіз: {'✅ УВІМКНЕНО' if self.use_mtf else '❌ ВИМКНЕНО'}")
        logger.info("=" * 60)

        # Завантаження моделей
        self.load_models()

        # Запуск WebSocket для real-time оновлень
        if self.enable_trading:
            logger.info("🔌 Запуск WebSocket...")
            await self.websocket.start()
            await asyncio.sleep(2)  # Дати час на підключення

        # Telegram старт
        await telegram_notifier.send_message(
            f"🚀 Trading Bot запущено\n"
            f"Символи: {', '.join(self.symbols)}\n"
            f"Режим: {'🔴 РЕАЛЬНІ УГОДИ' if self.enable_trading else '🟡 DEMO'}\n"
            f"Min confidence: {self.min_confidence:.0%}\n"
            f"Розмір позиції: ${self.position_size_usd} (leverage {self.leverage}x)\n"
            f"Margin на угоду: ${self.position_size_usd / self.leverage:.2f}\n"
            f"MTF аналіз: {'✅' if self.use_mtf else '❌'}\n"
            f"WebSocket: {'✅' if self.enable_trading else '❌'}\n"
            f"\n🛡️ ЗАХИСТ ВІД OVERTRADING:\n"
            f"• Cooldown після SL: {self.cooldown_after_sl / 60:.0f} хв\n"
            f"• Cooldown після TP: {self.cooldown_after_tp / 60:.0f} хв\n"
            f"• Max втрат/день: {self.max_daily_losses_per_symbol}"
        )

        # Баланс
        self.balance = await self.get_balance()

        iteration = 0

        try:
            while True:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Ітерація #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")

                try:
                    # Обробка кожного символу
                    for symbol in self.symbols:
                        if symbol not in self.models:
                            continue

                        logger.info(f"\n--- {symbol} ---")

                        # 🛡️ Перевірка blacklist status
                        if symbol in self.symbol_blacklist:
                            remaining = (self.symbol_blacklist[symbol] - datetime.now()).total_seconds() / 60
                            if remaining > 0:
                                logger.info(f"⏸️ В cooldown ще {remaining:.0f} хв")
                                continue

                        # Показати денний ліміт програшів
                        daily_losses = self.get_daily_losses_count(symbol)
                        if daily_losses > 0:
                            logger.info(f"📊 Програшів сьогодні: {daily_losses}/{self.max_daily_losses_per_symbol}")

                        # 1. MTF Прогноз (або стандартний якщо MTF вимкнено)
                        prediction = await self.predict_mtf(symbol)
                        if not prediction:
                            logger.warning(f"⚠️ {symbol}: прогноз не вдався")
                            continue

                        # 2. Перевірка позиції
                        position = await self.check_position(symbol)

                        if position:
                            logger.info(f"📊 Позиція: {position['side']} {abs(position['amount']):.6f} @ ${position['entry_price']:.2f}")
                            logger.info(f"💰 P&L: ${position['unrealized_pnl']:.2f}")
                        else:
                            logger.info("ℹ️ Позицій немає")

                        # 4. Торгова логіка
                        current_price = prediction['current_price']

                        if prediction['confidence'] >= self.min_confidence:
                            # UP сигнал
                            if prediction['prediction'] == 'UP':
                                if not position:
                                    logger.info(f"📈 СИГНАЛ BUY LONG (confidence: {prediction['confidence']:.2%})")
                                    await self.open_long_position(symbol, current_price, prediction['confidence'])
                                elif position['side'] == 'SHORT':
                                    logger.info(f"📈 СИГНАЛ CLOSE SHORT (confidence: {prediction['confidence']:.2%})")
                                    await self.close_short_position(symbol, position, current_price, f"ML Signal (conf: {prediction['confidence']:.2%})")

                            # DOWN сигнал
                            elif prediction['prediction'] == 'DOWN':
                                if not position:
                                    logger.info(f"📉 СИГНАЛ SELL SHORT (confidence: {prediction['confidence']:.2%})")
                                    await self.open_short_position(symbol, current_price, prediction['confidence'])
                                elif position['side'] == 'LONG':
                                    logger.info(f"📉 СИГНАЛ CLOSE LONG (confidence: {prediction['confidence']:.2%})")
                                    await self.close_long_position(symbol, position, current_price, f"ML Signal (conf: {prediction['confidence']:.2%})")
                        else:
                            logger.info(f"⏸️ HOLD (низька впевненість: {prediction['confidence']:.2%})")

                    # Оновлення балансу кожні 5 ітерацій
                    if iteration % 5 == 0:
                        self.balance = await self.get_balance()

                except Exception as e:
                    logger.error(f"❌ Помилка в ітерації: {e}", exc_info=True)

                # Очікування
                logger.info(f"⏳ Очікування {interval_seconds}с до наступної перевірки...")
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n🛑 Зупинка бота (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Критична помилка: {e}", exc_info=True)
        finally:
            # Зупинка WebSocket
            if self.enable_trading and self.websocket.is_running:
                logger.info("🔌 Зупинка WebSocket...")
                await self.websocket.stop()

            logger.info("👋 Trading bot зупинено")


async def main():
    """Головна функція"""
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Symbol Trading Bot (Random Forest)")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], help='Trading pairs (space-separated)')
    parser.add_argument('--testnet', action='store_true', default=True, help='Use testnet')
    parser.add_argument('--enable-trading', action='store_true', help='Enable real trading (default: demo)')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval (seconds)')
    parser.add_argument('--position-size', type=float, default=50.0, help='Position size in USD (default: 50.0)')
    parser.add_argument('--min-confidence', type=float, default=0.67, help='Minimum confidence (default: 0.67)')

    args = parser.parse_args()

    bot = SimpleTradingBot(
        symbols=args.symbols,
        testnet=args.testnet,
        enable_trading=args.enable_trading
    )
    bot.min_confidence = args.min_confidence
    bot.position_size_usd = args.position_size

    await bot.run(interval_seconds=args.interval)


if __name__ == '__main__':
    asyncio.run(main())