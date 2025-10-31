"""
Trading Bot - головний клас торгової системи
"""
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import logging
from binance.client import Client

from config import TRADING_CONFIG, BINANCE_CONFIG, MTF_CONFIG, WEBSOCKET_CONFIG, STRATEGY_CONFIG
from websocket_manager import BinanceFuturesWebSocket
from .ml_predictor import MLPredictor
from .position_manager import PositionManager, check_position_static
from .volatility_scanner import VolatilityScanner

logger = logging.getLogger(__name__)

# Нові стратегії
try:
    from strategies.strategy_selector import StrategySelector
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    logger.warning("⚠️ Стратегії не знайдені, використовую тільки ML")


class TradingBot:
    """Торговий бот з ML прогнозами та аналітикою"""
    
    def __init__(self):
        # Конфігурація
        self.symbols = TRADING_CONFIG['symbols']
        self.testnet = BINANCE_CONFIG['testnet']
        self.leverage = TRADING_CONFIG['leverage']
        self.position_size_usd = TRADING_CONFIG['position_size_usd']
        self.stop_loss_pct = TRADING_CONFIG['stop_loss_pct']
        self.take_profit_pct = TRADING_CONFIG['take_profit_pct']
        self.min_confidence = TRADING_CONFIG['min_confidence']
        self.max_positions = TRADING_CONFIG.get('max_positions', 10)  # Максимум одночасних позицій
        self.trading_config = TRADING_CONFIG  # 🔄 Зберігаємо для доступу до reverse settings
        
        # Binance клієнт з збільшеним timeout
        self.client = Client(
            BINANCE_CONFIG['api_key'],
            BINANCE_CONFIG['api_secret'],
            testnet=self.testnet,
            requests_params={'timeout': 30}  # 30 секунд замість 10
        )
        logger.info(f"✅ Binance client ({'TESTNET' if self.testnet else 'PRODUCTION'})")
        
        # ML прогнози (завжди з MTF - 15m+1h)
        self.predictor = MLPredictor(
            symbols=self.symbols,
            use_mtf=True  # Завжди використовуємо MTF (15m+1h)
        )
        
        # Нові стратегії (якщо доступні)
        self.use_strategies = STRATEGY_CONFIG.get('enabled', False) and STRATEGIES_AVAILABLE
        self.strategy_selector = None
        if self.use_strategies:
            self.strategy_selector = StrategySelector(config=STRATEGY_CONFIG)
            logger.info(f"✅ Strategy Selector активовано")
        else:
            logger.info(f"ℹ️ Використовую тільки ML моделі")
        
        # Управління позиціями
        self.position_manager = PositionManager(
            client=self.client,
            leverage=self.leverage,
            position_size_usd=self.position_size_usd,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct
        )
        
        # Волатильність сканер
        self.volatility_scanner = VolatilityScanner(
            client=self.client,
            symbols=self.symbols
        )
        self.min_volatility_score = TRADING_CONFIG.get('min_volatility_score', 25.0)  # Мінімальний скор волатильності
        
        # WebSocket
        self.websocket = BinanceFuturesWebSocket(self.client, testnet=self.testnet)
        self.websocket.on_order_update = self._on_order_update
        self.websocket.on_account_update = self._on_account_update
        
        # Захист від overtrading
        self.closed_positions = {}
        self.symbol_blacklist = {}
        self.force_close_blacklist = {}  # Окремий blacklist після force close
        self.cooldown_after_sl = TRADING_CONFIG['cooldown_after_sl']
        self.cooldown_after_tp = TRADING_CONFIG['cooldown_after_tp']
        self.cooldown_after_force_close = TRADING_CONFIG.get('cooldown_after_force_close', 7200)
        self.max_daily_losses_per_symbol = TRADING_CONFIG['max_daily_losses_per_symbol']
        
        # Статистика
        self.balance = 0.0
        
        # Встановлення leverage
        for symbol in self.symbols:
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
                logger.info(f"⚡ {symbol}: плече {self.leverage}x")
            except Exception as e:
                logger.warning(f"⚠️ Не вдалося встановити плече для {symbol}: {e}")
    
    def load_models(self):
        """Завантаження ML моделей"""
        self.predictor.load_models()
    
    async def get_market_data(self, symbol: str, interval: str = '4h', limit: int = 500) -> pd.DataFrame:
        """Завантаження ринкових даних з retry"""
        max_retries = 3
        retry_delay = 2  # секунди
        
        for attempt in range(max_retries):
            try:
                klines = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Спроба {attempt + 1}/{max_retries} для {symbol} {interval}: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"❌ Помилка завантаження даних {symbol} після {max_retries} спроб: {e}")
                    return pd.DataFrame()
    
    async def check_position(self, symbol: str) -> dict:
        """Перевірка позиції"""
        return await check_position_static(self.client, symbol)
    
    async def count_open_positions(self) -> int:
        """Підрахунок кількості відкритих позицій"""
        try:
            all_positions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.futures_position_information
            )
            
            count = 0
            for pos in all_positions:
                amt = float(pos['positionAmt'])
                if abs(amt) > 0.0001:
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"❌ Помилка підрахунку позицій: {e}")
            return 0
    
    def is_symbol_blacklisted(self, symbol: str) -> bool:
        """Перевірка blacklist"""
        # Перевірка звичайного blacklist
        if symbol in self.symbol_blacklist:
            if datetime.now() >= self.symbol_blacklist[symbol]:
                del self.symbol_blacklist[symbol]
            else:
                remaining = (self.symbol_blacklist[symbol] - datetime.now()).total_seconds() / 60
                logger.info(f"⏸️ {symbol}: cooldown {remaining:.0f} хв")
                return True
        
        # Перевірка force close blacklist (довший час)
        if symbol in self.force_close_blacklist:
            if datetime.now() >= self.force_close_blacklist[symbol]:
                del self.force_close_blacklist[symbol]
            else:
                remaining = (self.force_close_blacklist[symbol] - datetime.now()).total_seconds() / 60
                logger.warning(f"🚫 {symbol}: FORCE CLOSE blacklist {remaining:.0f} хв")
                return True
        
        return False
    
    def add_to_blacklist(self, symbol: str, reason: str, pnl: float):
        """Додати в blacklist"""
        # Для FORCE_CLOSE використовуємо окремий, довший cooldown
        if reason == 'FORCE_CLOSE':
            cooldown_seconds = self.cooldown_after_force_close
            self.force_close_blacklist[symbol] = datetime.now() + timedelta(seconds=cooldown_seconds)
            logger.error(
                f"🚫 {symbol}: FORCE CLOSE blacklist на {cooldown_seconds/60:.0f} хв "
                f"(PnL: ${pnl:+.2f}) - велика помилка прогнозу!"
            )
        else:
            cooldown_seconds = self.cooldown_after_sl if reason == 'SL' or pnl < 0 else self.cooldown_after_tp
            self.symbol_blacklist[symbol] = datetime.now() + timedelta(seconds=cooldown_seconds)
            logger.warning(f"🚫 {symbol}: blacklist на {cooldown_seconds/60:.0f} хв (reason: {reason}, PnL: ${pnl:+.2f})")
        
        if symbol not in self.closed_positions:
            self.closed_positions[symbol] = []
        
        self.closed_positions[symbol].append({
            'time': datetime.now(),
            'reason': reason,
            'pnl': pnl
        })
        
        logger.warning(f"🚫 {symbol}: blacklist на {cooldown_seconds/60:.0f} хв (reason: {reason}, PnL: ${pnl:+.2f})")
    
    def get_daily_losses_count(self, symbol: str) -> int:
        """Підрахунок втрат за день"""
        if symbol not in self.closed_positions:
            return 0
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        losses_today = [
            pos for pos in self.closed_positions[symbol]
            if pos['time'] >= today_start and pos['pnl'] < 0
        ]
        return len(losses_today)
    
    async def _on_order_update(self, order_info: dict):
        """WebSocket callback для ордерів"""
        try:
            symbol = order_info['symbol']
            status = order_info['status']
            order_type = order_info.get('order_type', '')
            
            if status == 'FILLED' and order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                position = await self.check_position(symbol)
                pnl = position['unrealized_pnl'] if position else 0.0
                
                if order_type == 'STOP_MARKET':
                    logger.warning(f"🛑 Stop-Loss: {symbol} (PnL: ${pnl:+.2f})")
                    self.add_to_blacklist(symbol, 'SL', pnl)
                else:
                    logger.info(f"🎯 Take-Profit: {symbol} (PnL: ${pnl:+.2f})")
                    self.add_to_blacklist(symbol, 'TP', pnl)
        except Exception as e:
            logger.error(f"❌ Помилка order update: {e}")
    
    async def _on_account_update(self, account_data: dict):
        """WebSocket callback для рахунку"""
        try:
            balances = account_data.get('B', [])
            for balance in balances:
                if balance['a'] == 'USDT':
                    new_balance = float(balance['wb'])
                    if abs(new_balance - self.balance) > 0.01:
                        self.balance = new_balance
        except Exception as e:
            logger.error(f"❌ Помилка account update: {e}")
    
    async def run_iteration(self):
        """Виконання однієї торгової ітерації"""
        try:
            logger.info("\n" + "="*80)
            logger.info("🔍 ФАЗА 1: Аналіз всіх пар")
            logger.info("="*80)
            
            # Збираємо всі сигнали
            all_signals = []
            positions_to_close = []
            positions_to_reverse = []  # 🔄 Нові розвороти
            
            for symbol in self.symbols:
                if symbol not in self.predictor.models:
                    continue
                
                logger.info(f"\n--- {symbol} ---")
                
                # Перевірка blacklist
                if self.is_symbol_blacklisted(symbol):
                    continue
                
                # Перевірка волатильності (мертвих пар)
                volatility_score = await self.volatility_scanner.calculate_volatility_score(symbol)
                if volatility_score < self.min_volatility_score:
                    logger.warning(f"🔇 {symbol}: низька волатильність {volatility_score:.1f} < {self.min_volatility_score:.1f} (мертва пара)")
                    continue
                else:
                    logger.info(f"📊 Волатильність: {volatility_score:.1f}/100")
                
                # Показати денні втрати
                daily_losses = self.get_daily_losses_count(symbol)
                if daily_losses >= self.max_daily_losses_per_symbol:
                    logger.warning(f"🚫 {symbol}: досягнуто ліміт втрат ({daily_losses}/{self.max_daily_losses_per_symbol})")
                    continue
                
                if daily_losses > 0:
                    logger.info(f"📊 Втрат сьогодні: {daily_losses}/{self.max_daily_losses_per_symbol}")
                
                # Прогноз: спочатку спробувати стратегії, потім ML
                prediction = None
                strategy_signal = None
                
                if self.use_strategies and self.strategy_selector:
                    # Спробувати нові стратегії
                    try:
                        df = await self.get_market_data(symbol, interval='4h', limit=500)
                        strategy_signal = self.strategy_selector.generate_signal(df, symbol)
                        
                        if strategy_signal:
                            # Конвертувати Signal в формат prediction
                            # LONG/SHORT → UP/DOWN для сумісності з ML
                            direction_map = {'LONG': 'UP', 'SHORT': 'DOWN', 'NEUTRAL': None}
                            ml_direction = direction_map.get(strategy_signal.direction)
                            
                            if ml_direction:
                                prediction = {
                                    'prediction': ml_direction,  # UP/DOWN для відкриття позицій
                                    'direction': ml_direction,   # Дублюємо для сумісності
                                    'confidence': strategy_signal.confidence / 100,  # 0-1
                                    'current_price': strategy_signal.entry_price,
                                    'metadata': strategy_signal.metadata,
                                    'source': 'strategy'
                                }
                                logger.info(f"🎯 {symbol}: Використано стратегію - {strategy_signal.reason}")
                    except Exception as e:
                        logger.error(f"❌ {symbol}: Помилка стратегії - {e}")
                
                # Fallback на ML якщо стратегії не дали сигналу
                if not prediction:
                    prediction = await self.predictor.predict_mtf(symbol, self.get_market_data)
                    if prediction:
                        prediction['source'] = 'ml'
                
                if not prediction:
                    logger.warning(f"⚠️ {symbol}: прогноз не вдався")
                    continue
                
                # Перевірка позиції
                position = await self.check_position(symbol)
                
                if position:
                    logger.info(f"📊 Позиція: {position['side']} {abs(position['amount']):.6f} @ ${position['entry_price']:.2f}")
                    logger.info(f"💰 P&L: ${position['unrealized_pnl']:.2f}")
                    
                    # Перевірка trailing stop
                    should_close_trailing = await self.position_manager.update_trailing_stop(
                        symbol, 
                        current_price=prediction['current_price'],
                        entry_price=position['entry_price'],
                        side=position['side']
                    )
                    
                    if should_close_trailing:
                        logger.info(f"🎯 TRAILING STOP: Позначено для закриття")
                        positions_to_close.append({
                            'symbol': symbol,
                            'position': position,
                            'price': prediction['current_price'],
                            'reason': 'Trailing Stop'
                        })
                        continue
                    
                    # Перевірка на розворот сигналу (🔄 СИЛЬНИЙ зворотний тренд)
                    reverse_enabled = self.trading_config.get('reverse_on_strong_signal', False)
                    reverse_min_conf = self.trading_config.get('reverse_min_confidence', 0.75)
                    reverse_profit_threshold = self.trading_config.get('reverse_profit_threshold', -0.005)
                    
                    # Розрахунок поточного P&L у відсотках
                    current_pnl_pct = position['unrealized_pnl'] / (self.position_size_usd / self.leverage)
                    
                    is_opposite_signal = (prediction['prediction'] == 'DOWN' and position['side'] == 'LONG') or \
                                        (prediction['prediction'] == 'UP' and position['side'] == 'SHORT')
                    
                    if is_opposite_signal and prediction['confidence'] >= self.min_confidence:
                        # РОЗВОРОТ: Закрити + відкрити зворотню
                        if reverse_enabled and \
                           prediction['confidence'] >= reverse_min_conf and \
                           current_pnl_pct < reverse_profit_threshold:
                            logger.warning(f"🔄 РОЗВОРОТ! Conf={prediction['confidence']:.1%}, PnL={current_pnl_pct:+.2%}")
                            positions_to_reverse.append({
                                'symbol': symbol,
                                'position': position,
                                'price': prediction['current_price'],
                                'new_direction': prediction['prediction'],
                                'confidence': prediction['confidence'],
                                'reason': f"Strong Reversal Signal (conf: {prediction['confidence']:.2%})"
                            })
                            continue
                        else:
                            # Звичайне закриття
                            positions_to_close.append({
                                'symbol': symbol,
                                'position': position,
                                'price': prediction['current_price'],
                                'reason': f"ML Signal Reversal (conf: {prediction['confidence']:.2%})"
                            })
                            continue
                else:
                    logger.info("ℹ️ Позицій немає")
                
                # Зберігаємо сигнали для нових позицій
                if not position and prediction['confidence'] >= self.min_confidence:
                    # Підтримка обох форматів: 'direction' (стратегії) та 'prediction' (ML)
                    pred_direction = prediction.get('direction') or prediction.get('prediction')
                    
                    all_signals.append({
                        'symbol': symbol,
                        'prediction': pred_direction,
                        'confidence': prediction['confidence'],
                        'current_price': prediction['current_price']
                    })
                    logger.info(f"✅ Сигнал: {pred_direction} (confidence: {prediction['confidence']:.2%})")
            
            # ФАЗА 2: Закриття позицій
            if positions_to_close:
                logger.info("\n" + "="*80)
                logger.info(f"🔄 ФАЗА 2: Закриття позицій ({len(positions_to_close)})")
                logger.info("="*80)
                
                for item in positions_to_close:
                    logger.info(f"\n📉 Закриття {item['symbol']} {item['position']['side']}")
                    if item['position']['side'] == 'LONG':
                        await self.position_manager.close_long(item['symbol'], item['position'], item['price'], item['reason'])
                    else:
                        await self.position_manager.close_short(item['symbol'], item['position'], item['price'], item['reason'])
            
            # ФАЗА 2.5: 🔄 РОЗВОРОТ позицій (закрити + відкрити зворотню)
            if positions_to_reverse:
                logger.info("\n" + "="*80)
                logger.info(f"🔄 ФАЗА 2.5: РОЗВОРОТ позицій ({len(positions_to_reverse)})")
                logger.info("="*80)
                
                for item in positions_to_reverse:
                    symbol = item['symbol']
                    position = item['position']
                    price = item['price']
                    new_direction = item['new_direction']
                    confidence = item['confidence']
                    
                    logger.info(f"\n🔄 {symbol}: {position['side']} → {new_direction} (conf: {confidence:.1%})")
                    
                    # 1. Закрити поточну
                    if position['side'] == 'LONG':
                        await self.position_manager.close_long(symbol, position, price, item['reason'])
                    else:
                        await self.position_manager.close_short(symbol, position, price, item['reason'])
                    
                    # Почекаємо трохи для виконання
                    await asyncio.sleep(0.5)
                    
                    # 2. Відкрити зворотню
                    try:
                        if new_direction == 'UP':
                            logger.info(f"📈 Відкриття LONG {symbol} @ ${price:.2f}")
                            await self.position_manager.open_long(symbol, price)
                        else:
                            logger.info(f"📉 Відкриття SHORT {symbol} @ ${price:.2f}")
                            await self.position_manager.open_short(symbol, price)
                    except Exception as e:
                        logger.error(f"❌ Помилка відкриття розвороту {symbol}: {e}")
            
            # ФАЗА 3: Відкриття нових позицій (топ за впевненістю)
            if all_signals:
                logger.info("\n" + "="*80)
                logger.info(f"🎯 ФАЗА 3: Відкриття нових позицій (знайдено {len(all_signals)} сигналів)")
                logger.info("="*80)
                
                # Сортуємо за впевненістю (від найбільшої до найменшої)
                all_signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Показуємо топ сигнали
                logger.info("\n📊 Рейтинг сигналів:")
                for i, signal in enumerate(all_signals[:10], 1):
                    logger.info(f"   {i}. {signal['symbol']}: {signal['prediction']} ({signal['confidence']:.2%})")
                
                # Перевіряємо скільки позицій можна відкрити
                open_positions_count = await self.count_open_positions()
                
                # 🔄 Резервуємо 1 слот для розвороту
                reserve_for_reversal = self.trading_config.get('reserve_slot_for_reversal', False)
                max_slots_to_use = self.max_positions - 1 if reserve_for_reversal else self.max_positions
                available_slots = max_slots_to_use - open_positions_count
                
                logger.info(f"\n💼 Відкритих позицій: {open_positions_count}/{self.max_positions}")
                if reserve_for_reversal:
                    logger.info(f"🔄 Резерв для розвороту: 1 слот")
                    logger.info(f"📈 Доступно для нових: {available_slots} (макс {max_slots_to_use})")
                else:
                    logger.info(f"📈 Доступно слотів: {available_slots}")
                
                if available_slots > 0:
                    # Відкриваємо топ N найкращих сигналів
                    signals_to_open = all_signals[:available_slots]
                    
                    logger.info(f"\n🚀 Відкриваємо {len(signals_to_open)} найкращих позицій:\n")
                    
                    for signal in signals_to_open:
                        logger.info(f"📈 ВІДКРИТТЯ: {signal['symbol']} {signal['prediction']} (confidence: {signal['confidence']:.2%})")
                        
                        if signal['prediction'] == 'UP':
                            await self.position_manager.open_long(
                                signal['symbol'], 
                                signal['current_price'], 
                                signal['confidence'], 
                                self.get_market_data
                            )
                        else:
                            await self.position_manager.open_short(
                                signal['symbol'], 
                                signal['current_price'], 
                                signal['confidence'], 
                                self.get_market_data
                            )
                        
                        # Маленька пауза між відкриттям позицій
                        await asyncio.sleep(0.5)
                else:
                    logger.warning(f"⚠️ Немає вільних слотів для нових позицій")
            else:
                logger.info("\n✅ Немає нових сигналів для відкриття позицій")
        
        except Exception as e:
            logger.error(f"❌ Помилка в ітерації: {e}", exc_info=True)
    
    async def ensure_all_positions_protected(self):
        """
        Перевірка та додавання SL/TP до всіх відкритих позицій без захисту
        Викликається при старті системи
        """
        logger.info("🛡️ Перевірка захисту всіх відкритих позицій...")
        
        try:
            # Отримання всіх позицій
            all_positions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.futures_position_information
            )
            
            positions_to_protect = []
            for pos in all_positions:
                amount = float(pos['positionAmt'])
                if abs(amount) > 0.0001:  # Є позиція
                    symbol = pos['symbol']
                    if symbol in self.symbols:  # Тільки наші символи
                        positions_to_protect.append(symbol)
            
            if not positions_to_protect:
                logger.info("✅ Немає відкритих позицій")
                return
            
            logger.info(f"🔍 Знайдено {len(positions_to_protect)} відкритих позицій: {', '.join(positions_to_protect)}")
            
            # Додавання захисту для кожної позиції
            protected_count = 0
            for symbol in positions_to_protect:
                success = await self.position_manager.ensure_position_protection(
                    symbol, 
                    self.get_market_data
                )
                if success:
                    protected_count += 1
                await asyncio.sleep(0.5)  # Невелика пауза між ордерами
            
            logger.info(f"✅ Захист додано для {protected_count}/{len(positions_to_protect)} позицій")
            
        except Exception as e:
            logger.error(f"❌ Помилка перевірки захисту позицій: {e}", exc_info=True)
