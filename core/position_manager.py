"""
Position Manager - управління позиціями (відкриття/закриття)
"""
import asyncio
from datetime import datetime
import logging
import pandas as pd

from telegram_bot import telegram_notifier
from training.online_learning import online_learner

logger = logging.getLogger(__name__)

# Database (optional)
try:
    from optimized.database.connection import DatabaseConnection
    from optimized.database.positions import PositionDB
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️ Database module not available")


# Статична функція для перевірки позиції
async def check_position_static(client, symbol: str) -> dict:
    """Перевірка позиції (статична версія) з retry"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.futures_position_information(symbol=symbol)
            )
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0.0001:
                        return {
                            'symbol': symbol,
                            'amount': amt,
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'side': 'LONG' if amt > 0 else 'SHORT'
                        }
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Спроба {attempt + 1}/{max_retries} перевірки позиції {symbol}: {e}")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Помилка перевірки позиції {symbol} після {max_retries} спроб: {e}")
                return None


class PositionManager:
    """Управління позиціями на Binance Futures"""
    
    def __init__(self, client, leverage: int, position_size_usd: float, 
                 stop_loss_pct: float, take_profit_pct: float):
        self.client = client
        self.leverage = leverage
        self.position_size_usd = position_size_usd
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Збереження даних для навчання
        self.open_positions_data = {}  # {symbol: {entry_time, prediction, confidence, features}}
        
        # Trailing stop для захисту прибутку
        self.trailing_stops = {}  # {symbol: {'best_price': float, 'trailing_stop': float}}
        
        # Кеш precision для кожного символу (отримуємо з exchangeInfo)
        self.symbol_info_cache = {}  # {symbol: {'quantityPrecision': int, 'pricePrecision': int, 'minQty': float}}
        
        # Database
        self.position_db = PositionDB() if DB_AVAILABLE else None
        if self.position_db:
            logger.info("✅ PositionDB initialized")
        else:
            logger.warning("⚠️ Trading without database")
    
    async def get_symbol_info(self, symbol: str) -> dict:
        """Отримання інформації про символ з exchangeInfo (з кешуванням)"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        try:
            exchange_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_exchange_info()
            )
            
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    # Знаходимо LOT_SIZE фільтр для мінімальної кількості
                    min_qty = 0.0
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            min_qty = float(f['minQty'])
                            break
                    
                    info = {
                        'quantityPrecision': s['quantityPrecision'],
                        'pricePrecision': s['pricePrecision'],
                        'minQty': min_qty
                    }
                    self.symbol_info_cache[symbol] = info
                    logger.info(f"📊 {symbol}: precision={info['quantityPrecision']}, minQty={min_qty}")
                    return info
            
            # Якщо не знайшли, повертаємо дефолтні значення
            logger.warning(f"⚠️ {symbol}: не знайдено в exchangeInfo, використовую дефолтні значення")
            return {'quantityPrecision': 2, 'pricePrecision': 2, 'minQty': 0.001}
            
        except Exception as e:
            logger.error(f"❌ Помилка отримання exchangeInfo для {symbol}: {e}")
            return {'quantityPrecision': 2, 'pricePrecision': 2, 'minQty': 0.001}
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Округлення quantity згідно правил Binance (використовує кеш)"""
        # Якщо є в кеші, використовуємо точну інформацію
        if symbol in self.symbol_info_cache:
            precision = self.symbol_info_cache[symbol]['quantityPrecision']
        else:
            # Fallback до оновлених значень (з test_exchange_info)
            precision_map = {
                'BTCUSDT': 3, 'ETHUSDT': 3, 'BNBUSDT': 2, 'SOLUSDT': 0,  # SOLUSDT: 0 (не 1)
                'ADAUSDT': 0, 'DOGEUSDT': 0, 'XRPUSDT': 1, 'LTCUSDT': 3,
                'LINKUSDT': 2, 'MATICUSDT': 0, 'DOTUSDT': 1, 'UNIUSDT': 2,
                'ATOMUSDT': 2, 'ETCUSDT': 2, 'XLMUSDT': 0, 'ALGOUSDT': 0,
                'VETUSDT': 0, 'FILUSDT': 1, 'TRXUSDT': 0, 'AVAXUSDT': 0  # AVAXUSDT: 0, FILUSDT: 1
            }
            precision = precision_map.get(symbol, 2)
        
        rounded = round(quantity, precision)
        return int(rounded) if precision == 0 else rounded
    
    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Форматування quantity для Binance API (використовує кеш)"""
        # Якщо є в кеші, використовуємо точну інформацію
        if symbol in self.symbol_info_cache:
            precision = self.symbol_info_cache[symbol]['quantityPrecision']
        else:
            # Fallback до оновлених значень (з test_exchange_info)
            precision_map = {
                'BTCUSDT': 3, 'ETHUSDT': 3, 'BNBUSDT': 2, 'SOLUSDT': 0,  # SOLUSDT: 0 (не 1)
                'ADAUSDT': 0, 'DOGEUSDT': 0, 'XRPUSDT': 1, 'LTCUSDT': 3,
                'LINKUSDT': 2, 'MATICUSDT': 0, 'DOTUSDT': 1, 'UNIUSDT': 2,
                'ATOMUSDT': 2, 'ETCUSDT': 2, 'XLMUSDT': 0, 'ALGOUSDT': 0,
                'VETUSDT': 0, 'FILUSDT': 1, 'TRXUSDT': 0, 'AVAXUSDT': 0  # AVAXUSDT: 0, FILUSDT: 1
            }
            precision = precision_map.get(symbol, 2)
        
        if precision == 0:
            return str(int(quantity))
        return f"{quantity:.{precision}f}"
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """Скасування всіх відкритих ордерів"""
        try:
            open_orders = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_get_open_orders(symbol=symbol)
            )
            
            if not open_orders:
                return True
            
            logger.info(f"🗑️ {symbol}: скасування {len(open_orders)} ордерів...")
            
            for order in open_orders:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda o=order: self.client.futures_cancel_order(
                            symbol=symbol,
                            orderId=o['orderId']
                        )
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Не вдалося скасувати ордер {order['orderId']}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Помилка скасування ордерів {symbol}: {e}")
            return False
    
    async def get_atr(self, symbol: str, get_market_data_func, period: int = 14) -> float:
        """Отримання ATR для динамічних SL/TP"""
        try:
            from training.rust_features import RustFeatureEngineer
            
            df = await get_market_data_func(symbol, '1h', 100)
            if df.empty:
                return 0.01
            
            feature_engineer = RustFeatureEngineer()
            df_features = feature_engineer.calculate_all(df, atr_periods=[period])
            
            atr_col = f'atr_{period}'
            if atr_col in df_features.columns:
                current_atr = df_features[atr_col].iloc[-1]
                if not pd.isna(current_atr):
                    return current_atr
            
            return 0.01
        except Exception as e:
            logger.error(f"❌ Помилка ATR для {symbol}: {e}")
            return 0.01
    
    async def update_trailing_stop(self, symbol: str, current_price: float, 
                                   entry_price: float, side: str) -> bool:
        """
        Оновлення trailing stop для захисту прибутку
        
        Args:
            symbol: Символ
            current_price: Поточна ціна
            entry_price: Ціна входу
            side: 'LONG' або 'SHORT'
        
        Returns:
            True якщо треба закривати (trailing stop спрацював)
        """
        try:
            # Відсоток прибутку
            if side == 'LONG':
                profit_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price
            
            # Активуємо trailing тільки коли є прибуток >= 1%
            if profit_pct < 0.01:
                return False
            
            # Ініціалізація trailing для символу
            if symbol not in self.trailing_stops:
                self.trailing_stops[symbol] = {
                    'best_profit_pct': profit_pct,
                    'best_price': current_price,
                    'activated': True
                }
                logger.info(f"🎯 {symbol}: Trailing stop активовано при +{profit_pct:.2%}")
                return False
            
            # Оновлення найкращої ціни
            trail = self.trailing_stops[symbol]
            if profit_pct > trail['best_profit_pct']:
                trail['best_profit_pct'] = profit_pct
                trail['best_price'] = current_price
                logger.info(f"📈 {symbol}: Новий пік прибутку +{profit_pct:.2%} @ ${current_price:.2f}")
            
            # Перевірка чи ціна відкотилася на 50% від піку
            # Якщо був +3%, закриваємо при +1.5%
            threshold_pct = trail['best_profit_pct'] * 0.5
            
            if profit_pct < threshold_pct:
                logger.info(
                    f"🛑 {symbol}: Trailing stop спрацював! "
                    f"Пік: +{trail['best_profit_pct']:.2%}, "
                    f"Зараз: +{profit_pct:.2%}, "
                    f"Поріг: +{threshold_pct:.2%}"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Помилка trailing stop {symbol}: {e}")
            return False
    
    async def open_long(self, symbol: str, price: float, confidence: float, get_market_data_func) -> bool:
        """Відкриття LONG позиції"""
        try:
            # Отримання інформації про символ (precision, minQty) - ВАЖЛИВО: робимо це ПЕРШИМ
            symbol_info = await self.get_symbol_info(symbol)
            
            # Скасувати старі ордери
            await self.cancel_all_orders(symbol)
            
            # Отримання актуальної ціни з біржі
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_symbol_ticker(symbol=symbol)
            )
            current_price = float(ticker['price'])
            
            # Розрахунок розміру позиції на основі актуальної ціни
            position_value = self.position_size_usd
            quantity = position_value / current_price
            
            # Округлення (використовує кеш з get_symbol_info)
            quantity = self.round_quantity(symbol, quantity)
            logger.info(f"📐 {symbol}: quantity={quantity} (precision={symbol_info['quantityPrecision']})")
            
            if quantity * current_price < 10:
                logger.warning(f"⚠️ Занадто маленька сума: ${quantity * current_price:.2f}")
                return False
            
            logger.info(f"📈 LONG {symbol}: {quantity} @ ${current_price:.2f} (target: ${self.position_size_usd})")
            
            # Виконання ордера
            quantity_str = self.format_quantity(symbol, quantity)
            
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
            
            # Очікування оновлення позиції
            await asyncio.sleep(1.5)
            
            # Отримання реальної ціни входу
            position_info = await check_position_static(self.client, symbol)
            
            if position_info and position_info['entry_price'] > 0:
                filled_price = position_info['entry_price']
                filled_qty = abs(position_info['amount'])
            else:
                filled_price = price
                filled_qty = quantity
            
            # SL/TP з конфігу (без ATR)
            sl_distance = filled_price * self.stop_loss_pct
            tp_distance = filled_price * self.take_profit_pct
            
            sl_price = filled_price - sl_distance
            tp_price = filled_price + tp_distance
            
            # Отримання price precision для правильного округлення
            price_precision = symbol_info.get('pricePrecision', 2)
            
            # Stop-loss
            try:
                sl_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='SELL',
                        type='STOP_MARKET',
                        stopPrice=round(sl_price, price_precision),
                        closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію, не відкривати SHORT!
                        timeInForce='GTC'  # 🔒 КРИТИЧНО: ордер не expired до скасування
                    )
                )
                logger.info(f"🛑 Stop-loss: ${sl_price:.{price_precision}f}")
            except Exception as e:
                logger.error(f"❌ Помилка SL: {e}")
            
            # Take-profit
            try:
                tp_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='SELL',
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=round(tp_price, price_precision),
                        closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію!
                        timeInForce='GTC'
                    )
                )
                logger.info(f"🎯 Take-profit: ${tp_price:.{price_precision}f}")
            except Exception as e:
                logger.error(f"❌ Помилка TP: {e}")
            
            # Telegram
            await telegram_notifier.send_message(
                f"✅ OPENED LONG\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${filled_price:.2f}\n"
                f"Quantity: {filled_qty:.6f}\n"
                f"Position: ${filled_qty * filled_price:.2f}\n"
                f"Margin: ${filled_qty * filled_price / self.leverage:.2f} ({self.leverage}x)\n"
                f"SL: ${sl_price:.2f}\n"
                f"TP: ${tp_price:.2f}\n"
                f"Confidence: {confidence:.2%}"
            )
            
            # БД - Збереження нової позиції
            if self.position_db:
                try:
                    position_id = await self.position_db.create_position(
                        symbol=symbol,
                        side='LONG',
                        entry_price=filled_price,
                        quantity=filled_qty,
                        leverage=self.leverage,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        ml_prediction='UP',
                        ml_confidence=confidence,
                        binance_order_id=order.get('orderId')
                    )
                    logger.info(f"💾 Position saved to DB: #{position_id}")
                except Exception as e:
                    logger.error(f"❌ БД помилка: {e}")
            
            # Збереження даних для online learning
            self.open_positions_data[symbol] = {
                'entry_time': datetime.now(),
                'entry_price': filled_price,
                'side': 'LONG',
                'prediction': 'UP',
                'confidence': confidence,
                'features': {}  # TODO: передавати фічі з ML predictor
            }
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Помилка відкриття LONG {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR opening LONG {symbol}: {str(e)[:100]}")
            return False
    
    async def open_short(self, symbol: str, price: float, confidence: float, get_market_data_func) -> bool:
        """Відкриття SHORT позиції"""
        try:
            # Отримання інформації про символ (precision, minQty) - ВАЖЛИВО: робимо це ПЕРШИМ
            symbol_info = await self.get_symbol_info(symbol)
            
            # Скасувати старі ордери
            await self.cancel_all_orders(symbol)
            
            # Отримання актуальної ціни з біржі
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_symbol_ticker(symbol=symbol)
            )
            current_price = float(ticker['price'])
            
            # Розрахунок розміру позиції на основі актуальної ціни
            position_value = self.position_size_usd
            quantity = position_value / current_price
            
            # Округлення (використовує кеш з get_symbol_info)
            quantity = self.round_quantity(symbol, quantity)
            logger.info(f"📐 {symbol}: quantity={quantity} (precision={symbol_info['quantityPrecision']})")
            
            if quantity * current_price < 10:
                logger.warning(f"⚠️ Занадто маленька сума: ${quantity * current_price:.2f}")
                return False
            
            logger.info(f"📉 SHORT {symbol}: {quantity} @ ${current_price:.2f} (target: ${self.position_size_usd})")
            
            # Виконання ордера
            quantity_str = self.format_quantity(symbol, quantity)
            
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
            
            await asyncio.sleep(1.5)
            
            # Отримання реальної ціни входу
            position_info = await check_position_static(self.client, symbol)
            
            if position_info and position_info['entry_price'] > 0:
                filled_price = position_info['entry_price']
                filled_qty = abs(position_info['amount'])
            else:
                filled_price = price
                filled_qty = quantity
            
            # SL/TP з конфігу (без ATR)
            sl_distance = filled_price * self.stop_loss_pct
            tp_distance = filled_price * self.take_profit_pct
            
            sl_price = filled_price + sl_distance  # SHORT: вище
            tp_price = filled_price - tp_distance  # SHORT: нижче
            
            # Отримання price precision для правильного округлення
            price_precision = symbol_info.get('pricePrecision', 2)
            
            # Stop-loss (BUY для закриття SHORT)
            try:
                sl_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='BUY',
                        type='STOP_MARKET',
                        stopPrice=round(sl_price, price_precision),
                        closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію, не відкривати LONG!
                        timeInForce='GTC'  # 🔒 КРИТИЧНО: ордер не expired до скасування
                    )
                )
                logger.info(f"🛑 Stop-loss: ${sl_price:.{price_precision}f}")
            except Exception as e:
                logger.error(f"❌ Помилка SL: {e}")
            
            # Take-profit (BUY для закриття SHORT)
            try:
                tp_order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side='BUY',
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=round(tp_price, price_precision),
                        closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію!
                        timeInForce='GTC'
                    )
                )
                logger.info(f"🎯 Take-profit: ${tp_price:.{price_precision}f}")
            except Exception as e:
                logger.error(f"❌ Помилка TP: {e}")
            
            # Telegram
            await telegram_notifier.send_message(
                f"📉 OPENED SHORT\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${filled_price:.2f}\n"
                f"Quantity: {filled_qty:.6f}\n"
                f"Position: ${filled_qty * filled_price:.2f}\n"
                f"Margin: ${filled_qty * filled_price / self.leverage:.2f} ({self.leverage}x)\n"
                f"SL: ${sl_price:.2f}\n"
                f"TP: ${tp_price:.2f}\n"
                f"Confidence: {confidence:.2%}"
            )
            
            # БД - Збереження нової позиції
            if self.position_db:
                try:
                    position_id = await self.position_db.create_position(
                        symbol=symbol,
                        side='SHORT',
                        entry_price=filled_price,
                        quantity=filled_qty,
                        leverage=self.leverage,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        ml_prediction='DOWN',
                        ml_confidence=confidence,
                        binance_order_id=order.get('orderId')
                    )
                    logger.info(f"💾 Position saved to DB: #{position_id}")
                except Exception as e:
                    logger.error(f"❌ БД помилка: {e}")
            
            # Збереження даних для online learning
            self.open_positions_data[symbol] = {
                'entry_time': datetime.now(),
                'entry_price': filled_price,
                'side': 'SHORT',
                'prediction': 'DOWN',
                'confidence': confidence,
                'features': {}  # TODO: передавати фічі з ML predictor
            }
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Помилка відкриття SHORT {symbol}: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ ERROR opening SHORT {symbol}: {str(e)[:100]}")
            return False
    
    async def close_long(self, symbol: str, position: dict, price: float, reason: str) -> bool:
        """Закриття LONG позиції"""
        try:
            quantity = abs(position['amount'])
            entry_price = position['entry_price']
            pnl = position['unrealized_pnl']
            
            logger.info(f"📉 Закриваємо LONG {symbol}: {quantity} @ ${price:.2f} (PnL: ${pnl:.2f})")
            
            quantity_str = self.format_quantity(symbol, quantity)
            
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity_str
                )
            )
            
            logger.info(f"✅ LONG закрито: {order['orderId']}")
            
            # PnL%
            price_change_pct = ((price - entry_price) / entry_price) * 100
            pnl_pct = price_change_pct * self.leverage
            
            emoji = "💰" if pnl > 0 else "📉"
            await telegram_notifier.send_message(
                f"{emoji} CLOSED LONG\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Exit: ${price:.2f}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                f"Reason: {reason}"
            )
            
            # БД - Закриття позиції
            if self.position_db:
                try:
                    # Знаходимо позицію в БД
                    db_position = await self.position_db.get_position_by_symbol(symbol)
                    if db_position:
                        await self.position_db.close_position(
                            position_id=db_position['id'],
                            exit_price=price,
                            exit_reason=reason,
                            realized_pnl=pnl,
                            realized_pnl_pct=pnl_pct
                        )
                except Exception as e:
                    logger.error(f"❌ БД помилка: {e}")
            
            # Online Learning - аналіз угоди
            if symbol in self.open_positions_data:
                position_data = self.open_positions_data[symbol]
                online_learner.analyze_closed_trade({
                    'symbol': symbol,
                    'side': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_time': position_data['entry_time'],
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'pnl_percentage': pnl_pct,
                    'exit_reason': reason,
                    'prediction': position_data['prediction'],
                    'confidence': position_data['confidence'],
                    'features': position_data.get('features', {})
                })
                del self.open_positions_data[symbol]
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Помилка закриття LONG {symbol}: {e}", exc_info=True)
            return False
    
    async def close_short(self, symbol: str, position: dict, price: float, reason: str) -> bool:
        """Закриття SHORT позиції"""
        try:
            quantity = abs(position['amount'])
            entry_price = position['entry_price']
            pnl = position['unrealized_pnl']
            
            logger.info(f"📈 Закриваємо SHORT {symbol}: {quantity} @ ${price:.2f} (PnL: ${pnl:.2f})")
            
            quantity_str = self.format_quantity(symbol, quantity)
            
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
            
            # PnL%
            price_change_pct = ((entry_price - price) / entry_price) * 100
            pnl_pct = price_change_pct * self.leverage
            
            await telegram_notifier.send_message(
                f"📈 CLOSED SHORT\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Exit: ${price:.2f}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                f"Reason: {reason}"
            )
            
            # БД - Закриття позиції
            if self.position_db:
                try:
                    # Знаходимо позицію в БД
                    db_position = await self.position_db.get_position_by_symbol(symbol)
                    if db_position:
                        await self.position_db.close_position(
                            position_id=db_position['id'],
                            exit_price=price,
                            exit_reason=reason,
                            realized_pnl=pnl,
                            realized_pnl_pct=pnl_pct
                        )
                except Exception as e:
                    logger.error(f"❌ БД помилка: {e}")
            
            # Online Learning - аналіз угоди
            if symbol in self.open_positions_data:
                position_data = self.open_positions_data[symbol]
                online_learner.analyze_closed_trade({
                    'symbol': symbol,
                    'side': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_time': position_data['entry_time'],
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'pnl_percentage': pnl_pct,
                    'exit_reason': reason,
                    'prediction': position_data['prediction'],
                    'confidence': position_data['confidence'],
                    'features': position_data.get('features', {})
                })
                del self.open_positions_data[symbol]
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Помилка закриття SHORT {symbol}: {e}", exc_info=True)
            return False
    
    async def ensure_position_protection(self, symbol: str, get_market_data_func) -> bool:
        """
        Перевірка та додавання SL/TP до існуючих позицій без захисту
        """
        try:
            # Перевірка позиції
            position = await check_position_static(self.client, symbol)
            if not position:
                return True  # Немає позиції
            
            # Перевірка наявних ордерів
            orders = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_get_open_orders(symbol=symbol)
            )
            
            has_sl = any(o['type'] == 'STOP_MARKET' for o in orders)
            has_tp = any(o['type'] == 'TAKE_PROFIT_MARKET' for o in orders)
            
            if has_sl and has_tp:
                return True  # Вже захищено
            
            logger.warning(f"🛡️ {symbol}: позиція без захисту (SL:{has_sl}, TP:{has_tp})")
            
            # Отримання параметрів позиції
            side = position['side']
            entry_price = position['entry_price']
            quantity = abs(position['amount'])
            
            # Розрахунок ATR для динамічних SL/TP
            atr = await self.get_atr(symbol, get_market_data_func, 14)
            sl_distance = 2.0 * atr
            tp_distance = 4.0 * atr
            
            # Для дуже дешевих монет (< $0.1) використовуємо мінімальну відстань у відсотках
            min_sl_pct = 0.015  # 1.5% мінімум для SL
            min_tp_pct = 0.03   # 3% мінімум для TP
            
            min_sl_distance = entry_price * min_sl_pct
            min_tp_distance = entry_price * min_tp_pct
            
            # Використовуємо більше значення (ATR або мінімальний відсоток)
            sl_distance = max(sl_distance, min_sl_distance)
            tp_distance = max(tp_distance, min_tp_distance)
            
            # Отримуємо інформацію про символ для precision
            symbol_info = await self.get_symbol_info(symbol)
            price_precision = symbol_info.get('pricePrecision', 2)
            
            if side == 'LONG':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
                close_side = 'SELL'
            else:  # SHORT
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
                close_side = 'BUY'
            
            # Додавання Stop-Loss
            if not has_sl:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type='STOP_MARKET',
                            stopPrice=round(sl_price, price_precision),
                            closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію!
                            timeInForce='GTC'  # 🔒 КРИТИЧНО: ордер не expired до скасування
                        )
                    )
                    logger.info(f"✅ {symbol}: додано SL @ ${sl_price:.{price_precision}f}")
                except Exception as e:
                    logger.error(f"❌ {symbol}: помилка SL - {e}")
            
            # Додавання Take-Profit
            if not has_tp:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type='TAKE_PROFIT_MARKET',
                            stopPrice=round(tp_price, price_precision),
                            closePosition=True,  # 🔥 КРИТИЧНО: ЗАКРИТИ позицію!
                            timeInForce='GTC'
                        )
                    )
                    logger.info(f"✅ {symbol}: додано TP @ ${tp_price:.{price_precision}f}")
                except Exception as e:
                    logger.error(f"❌ {symbol}: помилка TP - {e}")
            
            # Telegram notification
            if not (has_sl and has_tp):
                await telegram_notifier.send_message(
                    f"🛡️ ДОДАНО ЗАХИСТ\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"{'✅ SL додано' if not has_sl else '✓ SL є'}: ${sl_price:.2f}\n"
                    f"{'✅ TP додано' if not has_tp else '✓ TP є'}: ${tp_price:.2f}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Помилка додавання захисту {symbol}: {e}", exc_info=True)
            return False
