# -*- coding: utf-8 -*-
"""
Спрощений модуль торгової системи - ТІЛЬКИ BINANCE (без paper trading)
"""
import asyncio
import logging
import sys
import os
import argparse
import signal
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from intelligent_sys import UnifiedBinanceLoader, StrategyIntegration, create_strategy_integration
from intelligent_sys.utils import calculate_signal_confidence
from monitoring_system import monitoring_system
from cache_system import get_cache_info
from gpu_config import configure_gpu, get_gpu_info
from telegram_bot import telegram_notifier
from async_architecture import init_async_system, shutdown_async_system, ml_pipeline

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception
    BinanceRequestException = Exception

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if not BINANCE_AVAILABLE:
    raise RuntimeError("❌ python-binance не встановлено. Встановіть: pip install python-binance")


class SimpleTradingSystem:
    """Спрощена торгова система - тільки Binance API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        self.running = False
        self.shutdown_requested = False
        
        # Компоненти
        self.data_loader: Optional[UnifiedBinanceLoader] = None
        self.strategy_integration: Optional[StrategyIntegration] = None
        self.binance_client: Optional[Client] = None
        
        # Торгові дані
        self.symbols = config.get('symbols', [])
        self.portfolio_balance = 0.0
        self.positions: Dict[str, Dict] = {}
        self.ml_models: Dict[str, Any] = {}
        
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None

    async def initialize(self) -> bool:
        """Ініціалізація системи"""
        try:
            logger.info("🚀 Ініціалізація торгової системи (Binance only)...")
            
            load_dotenv()
            
            # Отримання API ключів
            self.api_key = os.getenv('FUTURES_API_KEY')
            self.api_secret = os.getenv('FUTURES_API_SECRET')
            
            if not self.api_key or not self.api_secret:
                raise RuntimeError("❌ FUTURES_API_KEY та FUTURES_API_SECRET обов'язкові!")
            
            logger.info(f"🔑 API ключ: {self.api_key[:4]}***{self.api_key[-4:]}")
            
            # GPU
            gpu_available = configure_gpu()
            if gpu_available:
                logger.info("✅ GPU доступний")
            
            # Async архітектура
            logger.info("🔧 Ініціалізація async системи...")
            await init_async_system()
            logger.info("✅ Async система готова")
            
            # Binance клієнт
            logger.info("🔌 Підключення до Binance...")
            use_testnet = os.getenv('USE_TESTNET', 'false').lower() in ('true', '1', 'yes')
            self.binance_client = Client(self.api_key, self.api_secret, testnet=use_testnet)
            
            if use_testnet:
                logger.info("✅ Binance TESTNET клієнт ініціалізовано")
            else:
                logger.info("✅ Binance PRODUCTION клієнт ініціалізовано")
            
            # Синхронізація балансу
            await self._sync_balance()
            
            # Data loader
            use_testnet = os.getenv('USE_TESTNET', 'false').lower() in ('true', '1', 'yes')
            self.data_loader = UnifiedBinanceLoader(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=use_testnet,
                use_public_data=False
            )
            logger.info(f"✅ Data loader готовий (testnet={use_testnet})")
            
            # Синхронізація даних
            if not self.config.get('skip_data_sync'):
                await self._sync_market_data()
            
            # Стратегії
            if self.config.get('enable_strategies'):
                logger.info("🧠 Ініціалізація стратегій...")
                self.strategy_integration = create_strategy_integration(
                    symbols=self.symbols,
                    portfolio_value=self.portfolio_balance,
                    enable_scalping=self.config.get('enable_scalping', False),
                    enable_day_trading=self.config.get('enable_day_trading', True),
                    enable_swing_trading=self.config.get('enable_swing_trading', True)
                )
                if self.strategy_integration and self.strategy_integration.initialized:
                    logger.info("✅ Стратегії готові")
                else:
                    raise RuntimeError("Не вдалося ініціалізувати стратегії")
            
            # Завантаження моделей
            await self._load_models()
            
            self.initialized = True
            self._print_stats()
            
            # Telegram повідомлення
            await telegram_notifier.send_system_status(
                status="запущена",
                details=f"Баланс: ${self.portfolio_balance:.2f}\nСимволи: {', '.join(self.symbols)}"
            )
            
            logger.info("✅ Система повністю готова")
            return True
            
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації: {e}", exc_info=True)
            return False
    
    async def _sync_balance(self):
        """Синхронізація балансу з Binance"""
        try:
            logger.info("💰 Синхронізація балансу...")
            account = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.binance_client.futures_account()
            )
            self.portfolio_balance = float(account['totalWalletBalance'])
            available = float(account.get('availableBalance', self.portfolio_balance))
            logger.info(f"✅ Баланс: ${self.portfolio_balance:.2f} USDT (доступно: ${available:.2f})")
            
            if self.strategy_integration:
                self.strategy_integration.update_portfolio_value(self.portfolio_balance)
                
        except Exception as e:
            logger.error(f"❌ Помилка синхронізації балансу: {e}")
            raise
    
    async def _sync_market_data(self):
        """Синхронізація ринкових даних"""
        from optimized_db import db_manager
        
        logger.info("📥 Синхронізація ринкових даних...")
        for symbol in self.symbols:
            try:
                saved = await self.data_loader.save_to_database(
                    db_manager, symbol, '1h', 7
                )
                logger.info(f"✅ {symbol}: {saved} записів")
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: {e}")
    
    async def _load_models(self):
        """Завантаження ML моделей"""
        logger.info("🤖 Завантаження ML моделей...")
        from optimized_model import OptimizedPricePredictionModel
        
        for symbol in self.symbols:
            model_path = f'models/{symbol}_best_model.h5'
            if os.path.exists(model_path):
                try:
                    model = OptimizedPricePredictionModel(
                        input_shape=(60, 20),
                        model_type='advanced_lstm'
                    )
                    model.load_model(model_path)
                    self.ml_models[symbol] = model
                    logger.info(f"✅ {symbol}: модель завантажена")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: {e}")
            else:
                logger.warning(f"⚠️ {symbol}: модель не знайдена")
    
    def _print_stats(self):
        """Виведення статистики"""
        logger.info("=" * 60)
        logger.info("📊 СТАТИСТИКА СИСТЕМИ")
        logger.info("=" * 60)
        logger.info(f"💰 Баланс: ${self.portfolio_balance:.2f}")
        logger.info(f"💹 Символи: {len(self.symbols)}")
        logger.info(f"🤖 Моделі: {len(self.ml_models)}/{len(self.symbols)}")
        if self.strategy_integration:
            perf = self.strategy_integration.get_performance_summary()
            logger.info(f"🎯 Стратегії: {perf.get('active_strategies', 0)}")
        logger.info("=" * 60)
    
    async def run(self):
        """Головний торговий цикл"""
        if not self.initialized:
            logger.error("❌ Система не ініціалізована")
            return
        
        self.running = True
        logger.info("🎯 Запуск торгового циклу...")
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Ітерація #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                try:
                    # 1. Синхронізація балансу
                    await self._sync_balance()
                    
                    # 2. Завантаження даних
                    logger.info("📊 Завантаження ринкових даних...")
                    market_data = await self.data_loader.get_multiple_symbols(
                        symbols=self.symbols,
                        interval='1h',
                        days_back=30
                    )
                    logger.info(f"✅ Завантажено дані для {len(market_data)} символів")
                    
                    # 3. ML прогнози
                    logger.info("🤖 Генерація прогнозів...")
                    predictions = await self._generate_predictions(market_data)
                    
                    # Відправка прогнозів у Telegram (кожні 5 ітерацій)
                    if iteration % 5 == 1:
                        await self._send_predictions_summary(predictions)
                    
                    # 4. Торгові сигнали
                    if self.strategy_integration:
                        logger.info("📈 Генерація сигналів...")
                        signals = await self.strategy_integration.generate_signals(
                            market_data=market_data,
                            predictions=predictions
                        )
                        
                        if signals:
                            logger.info(f"📊 Отримано {len(signals)} сигналів")
                            await self._execute_signals(signals, market_data)
                        else:
                            logger.info("ℹ️ Немає сигналів")
                    
                    # 5. Перевірка позицій
                    if self.positions:
                        await self._check_positions(market_data)
                    
                    # 6. Статус
                    if iteration % 5 == 0:
                        self._print_trading_status()
                    
                    # Завершення якщо --once
                    if self.config.get('run_once'):
                        logger.info("🛑 Режим --once: завершення")
                        break
                    
                    # Очікування
                    interval = self.config.get('trading_interval', 300)
                    logger.info(f"⏳ Очікування {interval}с...")
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"❌ Помилка в ітерації: {e}", exc_info=True)
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("🛑 Цикл скасовано")
        finally:
            self.running = False
            logger.info("🔴 Торговий цикл зупинено")
    
    async def _generate_predictions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Генерація ML прогнозів"""
        predictions = {}
        
        for symbol, df in market_data.items():
            if df.empty or len(df) < 60:
                continue
            
            try:
                current_price = df['close'].iloc[-1]
                
                # Якщо є модель
                if symbol in self.ml_models:
                    model = self.ml_models[symbol]
                    
                    # Підготовка даних
                    from optimized_indicators import OptimizedIndicatorCalculator
                    indicator_calc = OptimizedIndicatorCalculator()
                    indicators_dict = await indicator_calc.calculate_all_indicators_batch(df)
                    
                    # Додавання індикаторів
                    df['rsi'] = indicators_dict.get('RSI', pd.Series(index=df.index, dtype=float))
                    df['macd'] = indicators_dict.get('MACD', pd.Series(index=df.index, dtype=float))
                    df['macd_signal'] = indicators_dict.get('MACD_Signal', pd.Series(index=df.index, dtype=float))
                    df['bb_upper'] = indicators_dict.get('BB_Upper', pd.Series(index=df.index, dtype=float))
                    df['bb_lower'] = indicators_dict.get('BB_Lower', pd.Series(index=df.index, dtype=float))
                    df['bb_middle'] = df['close'].rolling(window=20).mean()
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    df['ema_12'] = df['close'].ewm(span=12).mean()
                    df['ema_26'] = df['close'].ewm(span=26).mean()
                    df['volume_sma'] = df['volume'].rolling(window=20).mean()
                    df['atr'] = indicators_dict.get('ATR', pd.Series(index=df.index, dtype=float))
                    df['adx'] = pd.Series(index=df.index, dtype=float).fillna(50)
                    df['stoch_k'] = indicators_dict.get('Stoch_K', pd.Series(index=df.index, dtype=float))
                    df['stoch_d'] = indicators_dict.get('Stoch_D', pd.Series(index=df.index, dtype=float))
                    df['cci'] = indicators_dict.get('CCI', pd.Series(index=df.index, dtype=float))
                    df['mfi'] = pd.Series(index=df.index, dtype=float).fillna(50)
                    df['willr'] = indicators_dict.get('Williams_R', pd.Series(index=df.index, dtype=float))
                    df['roc'] = indicators_dict.get('ROC', pd.Series(index=df.index, dtype=float))
                    df['obv'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
                    
                    feature_columns = [
                        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                        'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volume_sma',
                        'atr', 'adx', 'stoch_k', 'stoch_d', 'cci', 'mfi',
                        'willr', 'roc', 'obv'
                    ]
                    
                    df_clean = df.dropna()
                    
                    if len(df_clean) >= 60:
                        X = df_clean[feature_columns].values[-60:]
                        
                        # Нормалізація
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_seq = X_scaled.reshape(1, 60, len(feature_columns))
                        
                        # Прогноз
                        predicted_scaled = model.predict(X_seq)
                        predicted_price = current_price * (1 + (predicted_scaled[0] * 0.05))
                        predicted_change = (predicted_price - current_price) / current_price
                        
                        confidence = calculate_signal_confidence(predicted_change, df_clean)
                        
                        predictions[symbol] = {
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'predicted_change': predicted_change,
                            'change_percent': predicted_change,  # Додаємо для сумісності зі стратегіями
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        }
                        
                        logger.info(f"🤖 {symbol}: ${current_price:.2f} → ${predicted_price:.2f} ({predicted_change:.2%})")
                        continue
                
                # Базовий прогноз
                price_change = df['close'].pct_change().iloc[-1]
                predicted_change = price_change * 1.1
                predicted_price = current_price * (1 + predicted_change)
                confidence = calculate_signal_confidence(predicted_change, df)
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predicted_change': predicted_change,
                    'change_percent': predicted_change,  # Додаємо для сумісності зі стратегіями
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"❌ Помилка прогнозу {symbol}: {e}")
        
        return predictions
    
    async def _send_predictions_summary(self, predictions: Dict[str, Dict]):
        """Відправка підсумку прогнозів у Telegram"""
        if not predictions:
            return
        
        msg_lines = ["📊 ML ПРОГНОЗИ:\n"]
        
        # Сортуємо за зміною ціни
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: abs(x[1].get('predicted_change', 0)),
            reverse=True
        )
        
        for symbol, pred in sorted_preds[:5]:  # Топ-5
            current = pred.get('current_price', 0)
            predicted = pred.get('predicted_price', 0)
            change_pct = pred.get('predicted_change', 0) * 100
            
            emoji = "📈" if change_pct > 0 else "📉"
            msg_lines.append(
                f"{emoji} {symbol}: ${current:.2f} → ${predicted:.2f} ({change_pct:+.2f}%)"
            )
        
        await telegram_notifier.send_message("\n".join(msg_lines))
    
    async def _execute_signals(self, signals: Dict[str, Any], market_data: Dict[str, pd.DataFrame]):
        """Виконання торгових сигналів"""
        from strategies.base import TradeAction
        
        for symbol, signal in signals.items():
            try:
                # Валідація
                is_valid, reason = self.strategy_integration.validate_signal(signal)
                if not is_valid:
                    logger.warning(f"⚠️ {symbol}: сигнал відхилено - {reason}")
                    continue
                
                # Розмір позиції
                quantity = self.strategy_integration.calculate_position_size(signal)
                if quantity <= 0:
                    logger.warning(f"⚠️ {symbol}: некоректний розмір позиції")
                    continue
                
                # Перевірка наявності позиції
                if signal.action == TradeAction.BUY and symbol in self.positions:
                    logger.info(f"ℹ️ {symbol}: позиція вже відкрита")
                    continue
                if signal.action == TradeAction.SELL and symbol not in self.positions:
                    logger.info(f"ℹ️ {symbol}: немає позиції для закриття")
                    continue
                
                logger.info(f"📈 {symbol}: {signal.action.value} qty={quantity:.6f} price={signal.entry_price:.2f}")
                
                # Telegram
                await telegram_notifier.send_trade_signal(
                    symbol=symbol,
                    action=signal.action.value,
                    quantity=quantity,
                    price=signal.entry_price,
                    confidence=signal.confidence
                )
                
                # Виконання
                if signal.action == TradeAction.BUY:
                    await self._execute_buy(symbol, signal, quantity)
                elif signal.action == TradeAction.SELL:
                    await self._execute_sell(symbol, signal, quantity)
                    
            except Exception as e:
                logger.error(f"❌ Помилка виконання {symbol}: {e}", exc_info=True)
    
    async def _execute_buy(self, symbol: str, signal: Any, quantity: float):
        """Виконання BUY ордера"""
        try:
            # Округлення
            quantity = self._round_quantity(symbol, quantity)
            
            # Розміщення ордера
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.binance_client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
            )
            
            order_id = order['orderId']
            logger.info(f"� Ордер {symbol} створено (ID: {order_id})")
            
            # Чекаємо виконання ордера (до 5 секунд)
            for attempt in range(10):
                await asyncio.sleep(0.5)
                order_status = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.binance_client.futures_get_order(symbol=symbol, orderId=order_id)
                )
                
                if order_status['status'] == 'FILLED':
                    executed_price = float(order_status.get('avgPrice', 0))
                    executed_qty = float(order_status.get('executedQty', 0))
                    
                    if executed_price > 0 and executed_qty > 0:
                        logger.info(f"✅ BUY {symbol}: {executed_qty:.6f} @ ${executed_price:.2f}")
                        break
                    else:
                        logger.warning(f"⚠️ Ордер виконаний, але дані некоректні: price={executed_price}, qty={executed_qty}")
                        # Використовуємо поточну ціну
                        executed_price = signal.entry_price
                        executed_qty = quantity
                        break
            else:
                # Таймаут - використовуємо дані з ордера
                logger.warning(f"⚠️ Таймаут очікування виконання {symbol}, використовую origQty")
                executed_price = signal.entry_price
                executed_qty = quantity
            
            logger.info(f"✅ BUY {symbol}: {executed_qty:.6f} @ ${executed_price:.2f}")
            
            # Збереження позиції
            self.positions[symbol] = {
                'side': 'BUY',
                'entry_price': executed_price,
                'quantity': executed_qty,
                'entry_time': datetime.now(),
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'order_id': order_id
            }
            
            # Stop Loss
            if signal.stop_loss:
                try:
                    sl_price = await self._round_price(symbol, signal.stop_loss)
                    sl_order = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.binance_client.futures_create_order(
                            symbol=symbol,
                            side='SELL',
                            type='STOP_MARKET',
                            quantity=executed_qty,
                            stopPrice=str(sl_price)
                        )
                    )
                    self.positions[symbol]['sl_order_id'] = sl_order['orderId']
                    logger.info(f"🛑 SL {symbol}: ${sl_price:.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ SL помилка: {e}")
            
            # Take Profit
            if signal.take_profit:
                try:
                    tp_price = await self._round_price(symbol, signal.take_profit)
                    tp_order = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.binance_client.futures_create_order(
                            symbol=symbol,
                            side='SELL',
                            type='TAKE_PROFIT_MARKET',
                            quantity=executed_qty,
                            stopPrice=str(tp_price)
                        )
                    )
                    self.positions[symbol]['tp_order_id'] = tp_order['orderId']
                    logger.info(f"🎯 TP {symbol}: ${tp_price:.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ TP помилка: {e}")
            
            # Збереження в БД
            from optimized_db import db_manager, save_position
            position_data = {
                'symbol': symbol,
                'side': 'LONG',
                'entry_price': executed_price,
                'quantity': executed_qty,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'strategy': signal.strategy_name or 'unknown',
                'status': 'open',
                'metadata': {
                    'order_id': order_id,
                    'confidence': signal.confidence
                }
            }
            position_id = await save_position(db_manager, position_data)
            if position_id:
                self.positions[symbol]['position_id'] = position_id
                logger.info(f"💾 Позиція збережена (ID: {position_id})")
            
            # Telegram
            cost = executed_price * executed_qty
            await telegram_notifier.send_trade_execution(
                symbol=symbol,
                action="BUY",
                quantity=executed_qty,
                price=executed_price,
                cost=cost,
                balance=self.portfolio_balance,
                is_paper_trading=False
            )
            
            # Оновлення балансу
            await self._sync_balance()
            
        except Exception as e:
            logger.error(f"❌ Помилка BUY {symbol}: {e}", exc_info=True)
    
    async def _execute_sell(self, symbol: str, signal: Any, quantity: float):
        """Виконання SELL ордера"""
        if symbol not in self.positions:
            logger.warning(f"⚠️ {symbol}: немає позиції")
            return
        
        try:
            position = self.positions[symbol]
            
            # Скасування SL/TP
            for order_type in ['sl_order_id', 'tp_order_id']:
                if order_id := position.get(order_type):
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.binance_client.futures_cancel_order(symbol=symbol, orderId=order_id)
                        )
                    except:
                        pass
            
            # Округлення
            quantity = min(quantity, position['quantity'])
            quantity = self._round_quantity(symbol, quantity)
            
            # Розміщення ордера
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.binance_client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
            )
            
            executed_price = float(order.get('avgPrice', signal.entry_price))
            executed_qty = float(order.get('executedQty', quantity))
            
            # PnL
            pnl = (executed_price - position['entry_price']) * executed_qty
            revenue = executed_price * executed_qty
            
            logger.info(f"✅ SELL {symbol}: {executed_qty:.6f} @ ${executed_price:.2f} (PnL: ${pnl:.2f})")
            
            # Оновлення БД
            position_id = position.get('position_id')
            if position_id:
                from optimized_db import db_manager
                async with db_manager.async_session_factory() as session:
                    from sqlalchemy import text
                    await session.execute(text('''
                        UPDATE positions 
                        SET status = 'closed', 
                            exit_price = :exit_price,
                            exit_time = NOW(),
                            pnl = :pnl
                        WHERE id = :position_id
                    '''), {
                        'exit_price': executed_price,
                        'pnl': pnl,
                        'position_id': position_id
                    })
                    await session.commit()
                logger.info(f"💾 Позиція закрита в БД (ID: {position_id})")
            
            # Видалення з пам'яті
            del self.positions[symbol]
            
            # Запис в strategy integration
            if self.strategy_integration:
                self.strategy_integration.record_trade(
                    symbol=symbol,
                    pnl=pnl,
                    strategy_name=position.get('strategy_name')
                )
            
            # Telegram
            await telegram_notifier.send_trade_execution(
                symbol=symbol,
                action="SELL",
                quantity=executed_qty,
                price=executed_price,
                cost=revenue,
                balance=self.portfolio_balance,
                is_paper_trading=False
            )
            
            # Оновлення балансу
            await self._sync_balance()
            
        except Exception as e:
            logger.error(f"❌ Помилка SELL {symbol}: {e}", exc_info=True)
    
    async def _check_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Перевірка відкритих позицій"""
        if not self.strategy_integration:
            return
        
        current_prices = {
            symbol: df['close'].iloc[-1]
            for symbol, df in market_data.items()
            if not df.empty
        }
        
        close_decisions = await self.strategy_integration.check_close_positions(
            current_prices=current_prices,
            market_data=market_data
        )
        
        for symbol, should_close in close_decisions.items():
            if should_close and symbol in self.positions:
                try:
                    position = self.positions[symbol]
                    current_price = current_prices.get(symbol)
                    
                    if current_price:
                        # Створюємо фейковий сигнал для закриття
                        from strategies.base import TradeAction, TradingSignal
                        close_signal = TradingSignal(
                            symbol=symbol,
                            action=TradeAction.SELL,
                            entry_price=current_price,
                            quantity=position['quantity'],
                            confidence=0.8,
                            strategy_name=position.get('strategy_name')
                        )
                        
                        await self._execute_sell(symbol, close_signal, position['quantity'])
                        
                except Exception as e:
                    logger.error(f"❌ Помилка закриття {symbol}: {e}")
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Округлення кількості згідно з LOT_SIZE"""
        try:
            info = self.binance_client.futures_exchange_info()
            symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                # Шукаємо LOT_SIZE filter
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    step_size = float(lot_size_filter['stepSize'])
                    precision = len(str(step_size).rstrip('0').split('.')[-1])
                    rounded = round(quantity / step_size) * step_size
                    return round(rounded, precision)
            return quantity
        except Exception as e:
            logger.warning(f"⚠️ Помилка округлення {symbol}: {e}")
            return quantity
    
    async def _round_price(self, symbol: str, price: float) -> float:
        """Округлення ціни згідно з PRICE_FILTER"""
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.binance_client.futures_exchange_info()
            )
            symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                # Шукаємо PRICE_FILTER
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    precision = len(str(tick_size).rstrip('0').split('.')[-1])
                    rounded = round(price / tick_size) * tick_size
                    return round(rounded, precision)
            return price
        except Exception as e:
            logger.warning(f"⚠️ Помилка округлення ціни {symbol}: {e}")
            return price
    
    def _print_trading_status(self):
        """Виведення торгового статусу"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TRADING STATUS")
        logger.info("=" * 60)
        logger.info(f"💰 Баланс: ${self.portfolio_balance:.2f}")
        logger.info(f"📈 Позиції: {len(self.positions)}")
        
        if self.positions:
            for symbol, pos in self.positions.items():
                logger.info(f"  • {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.2f}")
        
        if self.strategy_integration:
            perf = self.strategy_integration.get_performance_summary()
            logger.info(f"📊 Угод: {perf.get('total_trades', 0)}")
            logger.info(f"✅ Win rate: {perf.get('win_rate', 0):.1%}")
            logger.info(f"💵 PnL: ${perf.get('total_pnl', 0):.2f}")
        
        logger.info("=" * 60 + "\n")
    
    async def shutdown(self):
        """Завершення роботи"""
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        
        logger.info("🔄 Завершення роботи...")
        
        self.running = False
        
        # Закриття компонентів
        if self.data_loader:
            try:
                await self.data_loader.close()
            except:
                pass
        
        if self.strategy_integration:
            try:
                self.strategy_integration.shutdown()
            except:
                pass
        
        try:
            await shutdown_async_system()
        except:
            pass
        
        logger.info("✅ Система завершена")


def setup_signal_handlers(system: SimpleTradingSystem, loop):
    """Налаштування обробників сигналів"""
    def handle_shutdown(signum, frame):
        logger.info(f"📡 Отримано сигнал {signal.Signals(signum).name}")
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(system.shutdown())
        )
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description="Торгова система (Binance only)")
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT',
                               'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'],
                       help='Список символів')
    parser.add_argument('--interval', type=int, default=300, help='Інтервал торгівлі (сек)')
    parser.add_argument('--once', action='store_true', help='Одна ітерація')
    parser.add_argument('--skip-data-sync', action='store_true', help='Пропустити синхронізацію даних')
    parser.add_argument('--enable-strategies', action='store_true', default=True, help='Увімкнути стратегії')
    parser.add_argument('--enable-scalping', action='store_true', help='Скальпінг')
    parser.add_argument('--enable-day-trading', action='store_true', default=True, help='Денна торгівля')
    parser.add_argument('--enable-swing-trading', action='store_true', default=True, help='Свінг-трейдинг')
    
    args = parser.parse_args()
    
    config = {
        'symbols': args.symbols,
        'trading_interval': args.interval,
        'run_once': args.once,
        'skip_data_sync': args.skip_data_sync,
        'enable_strategies': args.enable_strategies,
        'enable_scalping': args.enable_scalping,
        'enable_day_trading': args.enable_day_trading,
        'enable_swing_trading': args.enable_swing_trading
    }
    
    system = SimpleTradingSystem(config)
    loop = asyncio.get_running_loop()
    setup_signal_handlers(system, loop)
    
    try:
        if not await system.initialize():
            logger.error("❌ Помилка ініціалізації")
            return 1
        
        logger.info("🚀 Запуск системи...")
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl+C")
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}", exc_info=True)
        return 1
    finally:
        await system.shutdown()
    
    return 0


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        logger.info("✅ uvloop")
    except ImportError:
        pass
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
