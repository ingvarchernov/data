#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Синхронізація та контроль угод
Синхронізує базу даних з реальним станом на Binance при старті
"""
import asyncio
import logging
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Database
try:
    from optimized.database.connection import DatabaseConnection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️ База даних недоступна")


class TradesSynchronizer:
    """Синхронізація угод між Binance та базою даних"""
    
    def __init__(self, client: Client):
        self.client = client
        self.db = DatabaseConnection() if DB_AVAILABLE else None
    
    async def sync_on_startup(self):
        """Синхронізація при старті бота"""
        logger.info("🔄 Початок синхронізації з Binance...")
        
        try:
            # 1. Отримуємо всі відкриті позиції з Binance
            binance_positions = await self._get_open_positions_from_binance()
            logger.info(f"📊 Binance: {len(binance_positions)} відкритих позицій")
            
            # 2. Отримуємо всі відкриті позиції з БД
            db_positions = await self._get_open_positions_from_db()
            logger.info(f"💾 База даних: {len(db_positions)} відкритих позицій")
            
            # 3. Знаходимо закриті позиції (є в БД, але немає в Binance)
            db_symbols = {p['symbol'] for p in db_positions}
            binance_symbols = {p['symbol'] for p in binance_positions}
            
            closed_symbols = db_symbols - binance_symbols
            
            if closed_symbols:
                logger.info(f"🔍 Знайдено {len(closed_symbols)} закритих позицій: {', '.join(closed_symbols)}")
                
                # 4. Отримуємо дані про закриті угоди з Binance
                await self._sync_closed_positions(closed_symbols)
            else:
                logger.info("✅ Всі позиції в БД синхронізовані з Binance")
            
            # 5. Оновлюємо поточні ціни та unrealized PnL
            await self._update_unrealized_pnl(binance_positions)
            
            logger.info("✅ Синхронізація завершена")
            
        except Exception as e:
            logger.error(f"❌ Помилка синхронізації: {e}")
    
    async def _get_open_positions_from_binance(self):
        """Отримати відкриті позиції з Binance"""
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.futures_position_information
            )
            
            open_positions = []
            for pos in positions:
                amt = float(pos['positionAmt'])
                if abs(amt) > 0.0001:
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': 'LONG' if amt > 0 else 'SHORT',
                        'amount': abs(amt),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'leverage': int(pos['leverage'])
                    })
            
            return open_positions
            
        except Exception as e:
            logger.error(f"❌ Помилка отримання позицій з Binance: {e}")
            return []
    
    async def _get_open_positions_from_db(self):
        """Отримати відкриті позиції з БД"""
        if not self.db:
            return []
        
        try:
            async with self.db.async_engine.begin() as conn:
                result = await conn.execute(
                    """
                    SELECT symbol, side, entry_price, quantity, entry_time
                    FROM positions
                    WHERE status = 'open'
                    ORDER BY entry_time DESC
                    """
                )
                
                rows = result.fetchall()
                return [
                    {
                        'symbol': row[0],
                        'side': row[1],
                        'entry_price': float(row[2]),
                        'quantity': float(row[3]),
                        'entry_time': row[4]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"❌ Помилка отримання позицій з БД: {e}")
            return []
    
    async def _sync_closed_positions(self, closed_symbols):
        """Синхронізувати закриті позиції"""
        if not self.db:
            return
        
        for symbol in closed_symbols:
            try:
                # Отримуємо історію угод з Binance
                trades = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_account_trades(symbol=symbol, limit=20)
                )
                
                if not trades:
                    logger.warning(f"⚠️ {symbol}: немає історії угод")
                    continue
                
                # Знаходимо останню закриваючу угоду
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                total_realized_pnl = 0
                exit_time = None
                exit_price = None
                
                for trade in reversed(trades):  # Від нових до старих
                    trade_time = datetime.fromtimestamp(trade['time'] / 1000)
                    
                    if trade_time >= today:
                        realized_pnl = float(trade['realizedPnl'])
                        total_realized_pnl += realized_pnl
                        
                        if realized_pnl != 0 and exit_time is None:
                            exit_time = trade_time
                            exit_price = float(trade['price'])
                
                if exit_time:
                    # Оновлюємо позицію в БД
                    async with self.db.async_engine.begin() as conn:
                        # Отримуємо дані про позицію
                        result = await conn.execute(
                            """
                            SELECT id, entry_price, quantity, side
                            FROM positions
                            WHERE symbol = %s AND status = 'open'
                            ORDER BY entry_time DESC
                            LIMIT 1
                            """,
                            (symbol,)
                        )
                        
                        row = result.fetchone()
                        if row:
                            position_id = row[0]
                            entry_price = float(row[1])
                            quantity = float(row[2])
                            side = row[3]
                            
                            # Розраховуємо PnL %
                            if side == 'LONG':
                                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                            else:
                                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                            
                            # Визначаємо причину закриття
                            if total_realized_pnl > 0:
                                exit_reason = 'TAKE_PROFIT'
                            elif pnl_pct < -1.5:
                                exit_reason = 'STOP_LOSS'
                            else:
                                exit_reason = 'TRAILING_STOP'
                            
                            # Оновлюємо позицію
                            await conn.execute(
                                """
                                UPDATE positions
                                SET status = 'closed',
                                    exit_price = %s,
                                    exit_time = %s,
                                    pnl = %s,
                                    pnl_pct = %s,
                                    exit_reason = %s
                                WHERE id = %s
                                """,
                                (exit_price, exit_time, total_realized_pnl, pnl_pct, exit_reason, position_id)
                            )
                            
                            # Додаємо в таблицю trades
                            await conn.execute(
                                """
                                INSERT INTO trades (
                                    symbol, side, entry_price, exit_price,
                                    quantity, entry_time, exit_time,
                                    pnl, pnl_pct, exit_reason
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    symbol, side, entry_price, exit_price,
                                    quantity, row[4], exit_time,  # row[4] - entry_time
                                    total_realized_pnl, pnl_pct, exit_reason
                                )
                            )
                            
                            pnl_str = f"${total_realized_pnl:+.2f}" if total_realized_pnl != 0 else "$0.00"
                            logger.info(f"✅ {symbol}: закрито з PnL {pnl_str} ({pnl_pct:+.2f}%)")
                
            except Exception as e:
                logger.error(f"❌ Помилка синхронізації {symbol}: {e}")
    
    async def _update_unrealized_pnl(self, binance_positions):
        """Оновити unrealized PnL для відкритих позицій"""
        if not self.db:
            return
        
        for pos in binance_positions:
            try:
                async with self.db.async_engine.begin() as conn:
                    await conn.execute(
                        """
                        UPDATE positions
                        SET current_price = %s,
                            unrealized_pnl = %s,
                            unrealized_pnl_pct = %s,
                            updated_at = NOW()
                        WHERE symbol = %s AND status = 'open'
                        """,
                        (
                            pos['mark_price'],
                            pos['unrealized_pnl'],
                            (pos['unrealized_pnl'] / (pos['entry_price'] * pos['amount'])) * 100,
                            pos['symbol']
                        )
                    )
                    
            except Exception as e:
                logger.error(f"❌ Помилка оновлення PnL для {pos['symbol']}: {e}")
    
    async def close(self):
        """Закрити з'єднання"""
        if self.db:
            await self.db.close()


async def sync_trades_on_startup(client: Client):
    """Функція для виклику при старті бота"""
    syncer = TradesSynchronizer(client)
    await syncer.sync_on_startup()
    await syncer.close()
