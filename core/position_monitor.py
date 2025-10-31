#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 POSITION MONITOR - Моніторинг позицій в реальному часі
Критично важливий модуль для захисту від великих збитків
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from binance.client import Client

from config import TRADING_CONFIG
from telegram_bot import telegram_notifier

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Моніторинг відкритих позицій в реальному часі
    
    Функції:
    - Перевірка PnL кожні 30-60 секунд
    - Примусове закриття при великих збитках (>5%)
    - Автоматичне оновлення SL/TP
    - Виявлення застряглих позицій
    - Активація та оновлення trailing stop
    """
    
    def __init__(
        self,
        client: Client,
        position_manager,
        check_interval: int = 15,  # 🔥 КРИТИЧНО: 15 секунд замість 45 (3x швидше)
        max_loss_pct: float = 0.05,  # 5% максимальний збиток (125% на депозит)
        force_close_threshold: float = 0.04,  # 🚨 4% КРИТИЧНИЙ СТОП (100% на депозит = 1x втрата)
        stale_position_hours: int = 24,  # Позиція "застрягла" через 24 години
        on_force_close_callback = None  # Callback для blacklist після force close
    ):
        self.client = client
        self.position_manager = position_manager
        self.check_interval = check_interval
        self.max_loss_pct = max_loss_pct
        self.force_close_threshold = force_close_threshold
        self.stale_position_hours = stale_position_hours
        self.on_force_close_callback = on_force_close_callback
        
        # Налаштування trailing stop
        self.trailing_config = TRADING_CONFIG.get('trailing_stop', {})
        self.trailing_enabled = self.trailing_config.get('enabled', True)
        self.trailing_activation = self.trailing_config.get('activation_profit', 0.02)  # 2%
        self.trailing_distance = self.trailing_config.get('trail_distance', 0.30)  # 30%
        
        # Статистика
        self.monitor_stats = {
            'checks_count': 0,
            'force_closes': 0,
            'trailing_activations': 0,
            'trailing_closes': 0,
            'last_check': None
        }
        
        # Кеш попередніх цін для виявлення змін
        self.price_cache = {}
        
        logger.info("🔍 Position Monitor initialized")
        logger.info(f"   Check interval: {check_interval}s")
        logger.info(f"   Max loss: {max_loss_pct:.1%}")
        logger.info(f"   Force close: {force_close_threshold:.1%}")
        logger.info(f"   Trailing stop: {'✅' if self.trailing_enabled else '❌'}")
    
    async def monitor_loop(self):
        """Головний цикл моніторингу"""
        logger.info("🚀 Position Monitor started")
        
        while True:
            try:
                await self._check_all_positions()
                self.monitor_stats['checks_count'] += 1
                self.monitor_stats['last_check'] = datetime.now()
                
                # Чекаємо до наступної перевірки
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Коротка пауза при помилці
    
    async def _check_all_positions(self):
        """Перевірка всіх відкритих позицій"""
        try:
            # Отримуємо всі позиції з Binance
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.futures_position_information
            )
            
            open_positions = [
                pos for pos in positions
                if abs(float(pos['positionAmt'])) > 0.0001
            ]
            
            if not open_positions:
                return
            
            logger.debug(f"🔍 Checking {len(open_positions)} positions...")
            
            # Перевіряємо кожну позицію
            for pos in open_positions:
                await self._check_position(pos)
                
        except Exception as e:
            logger.error(f"❌ Error checking positions: {e}")
    
    async def _check_position(self, pos: dict):
        """Перевірка однієї позиції"""
        try:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            
            side = 'LONG' if amt > 0 else 'SHORT'
            
            # Розрахунок PnL у відсотках
            initial_margin = float(pos['positionInitialMargin'])
            if initial_margin <= 0:
                return
            
            pnl_pct = unrealized_pnl / initial_margin
            
            # Розрахунок зміни ціни
            if side == 'LONG':
                price_change_pct = (mark_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - mark_price) / entry_price
            
            # 🔴 КРИТИЧНИЙ ЗБИТОК - Примусове закриття
            if pnl_pct < -self.force_close_threshold:
                await self._force_close_position(
                    symbol, pos, pnl_pct,
                    reason="FORCE_CLOSE"  # Без деталей - для БД constraint
                )
                return
            
            # 🟡 ВЕЛИКИЙ ЗБИТОК - Попередження
            if pnl_pct < -self.max_loss_pct:
                logger.warning(
                    f"⚠️ {symbol} {side}: Large loss {pnl_pct:+.2%} "
                    f"(${unrealized_pnl:+.2f}) @ ${mark_price:.4f}"
                )
                
                # Перевіряємо чи не треба закрити раніше
                if pnl_pct < -self.max_loss_pct * 1.2:  # -6% при max_loss=5%
                    await telegram_notifier.send_message(
                        f"🚨 LARGE LOSS ALERT\n"
                        f"{symbol} {side}\n"
                        f"PnL: {pnl_pct:+.2%} (${unrealized_pnl:+.2f})\n"
                        f"Entry: ${entry_price:.4f}\n"
                        f"Mark: ${mark_price:.4f}"
                    )
            
            # 🟢 TRAILING STOP для прибуткових позицій
            if self.trailing_enabled and price_change_pct >= self.trailing_activation:
                await self._handle_trailing_stop(
                    symbol, pos, side, entry_price, mark_price, price_change_pct
                )
            
            # 🕐 ЗАСТРЯГЛА ПОЗИЦІЯ
            await self._check_stale_position(symbol, pos)
            
            # Оновлюємо кеш цін
            self.price_cache[symbol] = {
                'mark_price': mark_price,
                'pnl_pct': pnl_pct,
                'last_update': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking position {pos.get('symbol')}: {e}")
    
    async def _force_close_position(
        self,
        symbol: str,
        pos: dict,
        pnl_pct: float,
        reason: str
    ):
        """Примусове закриття позиції"""
        try:
            amt = float(pos['positionAmt'])
            side = 'LONG' if amt > 0 else 'SHORT'
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            
            logger.error(
                f"🚨 FORCE CLOSE: {symbol} {side}\n"
                f"   PnL: {pnl_pct:+.2%} (${unrealized_pnl:+.2f})\n"
                f"   Reason: {reason}"
            )
            
            # Закриваємо позицію
            position_data = {
                'symbol': symbol,
                'amount': abs(amt),
                'entry_price': float(pos['entryPrice']),
                'unrealized_pnl': unrealized_pnl,
                'side': side
            }
            
            if side == 'LONG':
                success = await self.position_manager.close_long(
                    symbol, position_data, mark_price, reason
                )
            else:
                success = await self.position_manager.close_short(
                    symbol, position_data, mark_price, reason
                )
            
            if success:
                self.monitor_stats['force_closes'] += 1
                
                # Викликаємо callback для додавання в blacklist
                if self.on_force_close_callback:
                    try:
                        self.on_force_close_callback(symbol, 'FORCE_CLOSE', unrealized_pnl)
                    except Exception as e:
                        logger.error(f"❌ Error in force close callback: {e}")
                
                # Запускаємо екстрене дотренування в фоні (не блокуємо)
                asyncio.create_task(self._trigger_emergency_retrain(symbol, pnl_pct))
                
                # Telegram повідомлення (без Markdown щоб уникнути parse errors)
                pnl_pct_str = f"{pnl_pct:+.2%}"
                pnl_usd_str = f"${unrealized_pnl:+.2f}"
                
                await telegram_notifier.send_message(
                    f"🚨 FORCE CLOSED\n"
                    f"====================\n"
                    f"{symbol} {side}\n"
                    f"PnL: {pnl_pct_str} ({pnl_usd_str})\n"
                    f"Critical loss detected\n"
                    f"\n⚠️ Position protection activated",
                    parse_mode=None  # Відключаємо Markdown
                )
        except Exception as e:
            logger.error(f"❌ Error force closing {symbol}: {e}")
    
    async def _handle_trailing_stop(
        self,
        symbol: str,
        pos: dict,
        side: str,
        entry_price: float,
        mark_price: float,
        profit_pct: float
    ):
        """Обробка trailing stop"""
        try:
            # Отримуємо дані trailing з position_manager
            if symbol not in self.position_manager.trailing_stops:
                # Активуємо trailing
                self.position_manager.trailing_stops[symbol] = {
                    'best_profit_pct': profit_pct,
                    'best_price': mark_price,
                    'activated': True
                }
                
                self.monitor_stats['trailing_activations'] += 1
                
                logger.info(
                    f"🎯 {symbol}: Trailing stop ACTIVATED at +{profit_pct:.2%} "
                    f"(${mark_price:.4f})"
                )
                return
            
            # Оновлюємо trailing
            trail = self.position_manager.trailing_stops[symbol]
            
            # Новий пік?
            if profit_pct > trail['best_profit_pct']:
                old_best = trail['best_profit_pct']
                trail['best_profit_pct'] = profit_pct
                trail['best_price'] = mark_price
                
                logger.info(
                    f"📈 {symbol}: New profit peak +{profit_pct:.2%} "
                    f"(was +{old_best:.2%})"
                )
            
            # Перевіряємо чи треба закривати
            threshold_pct = trail['best_profit_pct'] * (1 - self.trailing_distance)
            
            if profit_pct < threshold_pct:
                # Trailing stop спрацював!
                logger.info(
                    f"🛑 {symbol}: TRAILING STOP triggered!\n"
                    f"   Peak: +{trail['best_profit_pct']:.2%}\n"
                    f"   Current: +{profit_pct:.2%}\n"
                    f"   Threshold: +{threshold_pct:.2%}"
                )
                
                # Закриваємо
                amt = float(pos['positionAmt'])
                unrealized_pnl = float(pos['unRealizedProfit'])
                
                position_data = {
                    'symbol': symbol,
                    'amount': abs(amt),
                    'entry_price': entry_price,
                    'unrealized_pnl': unrealized_pnl,
                    'side': side
                }
                
                if side == 'LONG':
                    success = await self.position_manager.close_long(
                        symbol, position_data, mark_price, "TRAILING_STOP"
                    )
                else:
                    success = await self.position_manager.close_short(
                        symbol, position_data, mark_price, "TRAILING_STOP"
                    )
                
                if success:
                    self.monitor_stats['trailing_closes'] += 1
                    
                    # Видаляємо з trailing
                    del self.position_manager.trailing_stops[symbol]
                    
                    # Telegram
                    await telegram_notifier.send_message(
                        f"🎯 TRAILING STOP\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"{symbol} {side}\n"
                        f"Peak: +{trail['best_profit_pct']:.2%}\n"
                        f"Closed: +{profit_pct:.2%}\n"
                        f"PnL: ${unrealized_pnl:+.2f}\n"
                        f"\n✅ Profit secured"
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error handling trailing stop for {symbol}: {e}")
    
    async def _check_stale_position(self, symbol: str, pos: dict):
        """Перевірка на застряглу позицію"""
        try:
            # TODO: Додати перевірку часу відкриття з БД
            # Зараз просто логуємо якщо позиція довго без змін
            
            if symbol in self.price_cache:
                last_check = self.price_cache[symbol].get('last_update')
                if last_check and (datetime.now() - last_check).total_seconds() > 3600:
                    # Позиція не змінювалась більше години
                    mark_price = float(pos['markPrice'])
                    cached_price = self.price_cache[symbol]['mark_price']
                    
                    price_diff = abs(mark_price - cached_price) / cached_price
                    
                    if price_diff < 0.001:  # Менше 0.1% зміни
                        logger.warning(
                            f"⏰ {symbol}: Stale position detected "
                            f"(no significant price movement for 1h)"
                        )
        except Exception as e:
            logger.debug(f"Error checking stale position {symbol}: {e}")
    
    def get_stats(self) -> Dict:
        """Отримання статистики монітора"""
        return {
            **self.monitor_stats,
            'uptime_seconds': (
                (datetime.now() - self.monitor_stats['last_check']).total_seconds()
                if self.monitor_stats['last_check'] else 0
            )
        }
    
    async def _trigger_emergency_retrain(self, symbol: str, loss_pct: float):
        """
        Екстрене дотренування моделі після критичної помилки
        
        Запускається в фоні після force close, не блокує торгівлю
        """
        try:
            logger.warning(f"🔄 Запуск екстреного дотренування для {symbol} (втрата: {loss_pct:.2%})")
            
            from incremental_retrain import IncrementalRetrainer
            retrainer = IncrementalRetrainer()
            
            # Перевіряємо чи є достатньо невдалих угод
            failed_positions = await retrainer.get_failed_positions(symbol=symbol, days=1)
            
            if len(failed_positions) < 2:
                logger.info(f"ℹ️ {symbol}: недостатньо даних для дотренування (тільки {len(failed_positions)} позицій)")
                return
            
            logger.info(f"📊 {symbol}: знайдено {len(failed_positions)} невдалих позицій за останню добу")
            
            # Дотренування
            success = await retrainer.retrain_symbol(symbol, reason="emergency_force_close")
            
            if success:
                logger.info(f"✅ {symbol}: екстрене дотренування успішне")
                await telegram_notifier.send_message(
                    f"✅ EMERGENCY RETRAIN\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{symbol}\n"
                    f"Причина: Force close {loss_pct:.2%}\n"
                    f"Дані: {len(failed_positions)} невдалих позицій\n"
                    f"Статус: Успішно",
                    parse_mode=None
                )
            else:
                logger.error(f"❌ {symbol}: екстрене дотренування провалилось")
            
            # Закриття БД
            if hasattr(retrainer.db, 'db') and hasattr(retrainer.db.db, 'close'):
                await retrainer.db.db.close()
            
        except Exception as e:
            logger.error(f"❌ Помилка екстреного дотренування {symbol}: {e}")
            import traceback
            traceback.print_exc()


async def start_monitor(
    client: Client, 
    position_manager,
    on_force_close_callback=None
) -> PositionMonitor:
    """
    Запуск монітора позицій
    
    Args:
        client: Binance client
        position_manager: PositionManager instance
        on_force_close_callback: Callback(symbol, reason, pnl) для blacklist
    
    Usage:
        monitor = await start_monitor(client, position_manager, callback)
        # Моніторинг працює в фоні
    """
    monitor = PositionMonitor(
        client, 
        position_manager,
        on_force_close_callback=on_force_close_callback
    )
    
    # Запускаємо в background
    asyncio.create_task(monitor.monitor_loop())
    
    logger.info("✅ Position Monitor started in background")
    return monitor
