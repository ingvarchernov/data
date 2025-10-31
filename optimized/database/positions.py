#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 DATABASE OPERATIONS FOR POSITIONS
Операції з позиціями в базі даних
"""
import logging
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import text
import json

from optimized.database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class PositionDB:
    """Управління позиціями в базі даних"""
    
    def __init__(self, db: DatabaseConnection = None):
        if db is None:
            db = DatabaseConnection()
        self.db = db
    
    async def create_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss_price: float = None,
        take_profit_price: float = None,
        ml_prediction: str = None,
        ml_confidence: float = None,
        ml_features: dict = None,
        binance_order_id: int = None,
        session_id: int = None
    ) -> int:
        """
        Створення нової позиції
        
        Returns:
            position_id
        """
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    text("""
                        INSERT INTO positions (
                            session_id, symbol, side, status,
                            entry_price, quantity, leverage, entry_time,
                            stop_loss_price, take_profit_price,
                            ml_prediction, ml_confidence, ml_features,
                            binance_order_id
                        ) VALUES (
                            :session_id, :symbol, :side, 'open',
                            :entry_price, :quantity, :leverage, NOW(),
                            :stop_loss_price, :take_profit_price,
                            :ml_prediction, :ml_confidence, :ml_features,
                            :binance_order_id
                        )
                        RETURNING id
                    """),
                    {
                        'session_id': session_id,
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'leverage': leverage,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'ml_prediction': ml_prediction,
                        'ml_confidence': ml_confidence,
                        'ml_features': json.dumps(ml_features) if ml_features else None,
                        'binance_order_id': binance_order_id
                    }
                )
                position_id = result.scalar_one()
                await session.commit()
                
                logger.info(f"✅ Position created: #{position_id} {symbol} {side} @ ${entry_price}")
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='OPEN',
                    new_status='open',
                    current_price=entry_price,
                    reason=f"Opened {side} position",
                    triggered_by='SYSTEM'
                )
                
                return position_id
        except Exception as e:
            logger.error(f"❌ Error creating position: {e}")
            return None
    
    async def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str,
        realized_pnl: float = None,
        realized_pnl_pct: float = None,
        fees: float = 0.0
    ) -> bool:
        """Закриття позиції"""
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    text("""
                        UPDATE positions
                        SET 
                            status = 'closed',
                            exit_price = :exit_price,
                            exit_time = NOW(),
                            exit_reason = :exit_reason,
                            realized_pnl = :realized_pnl,
                            realized_pnl_pct = :realized_pnl_pct,
                            fees = :fees,
                            updated_at = NOW()
                        WHERE id = :position_id
                    """),
                    {
                        'position_id': position_id,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'realized_pnl': realized_pnl,
                        'realized_pnl_pct': realized_pnl_pct,
                        'fees': fees
                    }
                )
                await session.commit()
                
                logger.info(
                    f"✅ Position closed: #{position_id} @ ${exit_price} "
                    f"PnL: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)"
                )
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='CLOSE',
                    old_status='open',
                    new_status='closed',
                    current_price=exit_price,
                    unrealized_pnl=realized_pnl,
                    unrealized_pnl_pct=realized_pnl_pct,
                    reason=f"Closed: {exit_reason}",
                    triggered_by='SYSTEM'
                )
                
                return True
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False
    
    async def update_stop_loss(
        self,
        position_id: int,
        new_stop_loss: float,
        reason: str = "Updated SL"
    ) -> bool:
        """Оновлення stop loss"""
        try:
            async with self.db.async_session_factory() as session:
                # Отримуємо старе значення
                result = await session.execute(
                    text("SELECT stop_loss_price FROM positions WHERE id = :id"),
                    {'id': position_id}
                )
                old_sl = result.scalar_one_or_none()
                
                # Оновлюємо
                await session.execute(
                    text("""
                        UPDATE positions
                        SET stop_loss_price = :new_sl, updated_at = NOW()
                        WHERE id = :position_id
                    """),
                    {'position_id': position_id, 'new_sl': new_stop_loss}
                )
                await session.commit()
                
                logger.info(f"✅ Position #{position_id}: SL updated ${old_sl} → ${new_stop_loss}")
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='UPDATE_SL',
                    old_stop_loss=old_sl,
                    new_stop_loss=new_stop_loss,
                    reason=reason,
                    triggered_by='SYSTEM'
                )
                
                return True
        except Exception as e:
            logger.error(f"❌ Error updating stop loss: {e}")
            return False
    
    async def update_take_profit(
        self,
        position_id: int,
        new_take_profit: float,
        reason: str = "Updated TP"
    ) -> bool:
        """Оновлення take profit"""
        try:
            async with self.db.async_session_factory() as session:
                # Отримуємо старе значення
                result = await session.execute(
                    text("SELECT take_profit_price FROM positions WHERE id = :id"),
                    {'id': position_id}
                )
                old_tp = result.scalar_one_or_none()
                
                # Оновлюємо
                await session.execute(
                    text("""
                        UPDATE positions
                        SET take_profit_price = :new_tp, updated_at = NOW()
                        WHERE id = :position_id
                    """),
                    {'position_id': position_id, 'new_tp': new_take_profit}
                )
                await session.commit()
                
                logger.info(f"✅ Position #{position_id}: TP updated ${old_tp} → ${new_take_profit}")
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='UPDATE_TP',
                    old_take_profit=old_tp,
                    new_take_profit=new_take_profit,
                    reason=reason,
                    triggered_by='SYSTEM'
                )
                
                return True
        except Exception as e:
            logger.error(f"❌ Error updating take profit: {e}")
            return False
    
    async def activate_trailing_stop(
        self,
        position_id: int,
        trail_distance: float,
        best_price: float
    ) -> bool:
        """Активація trailing stop"""
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    text("""
                        UPDATE positions
                        SET 
                            trailing_stop_active = TRUE,
                            trailing_stop_distance = :trail_distance,
                            best_price = :best_price,
                            updated_at = NOW()
                        WHERE id = :position_id
                    """),
                    {
                        'position_id': position_id,
                        'trail_distance': trail_distance,
                        'best_price': best_price
                    }
                )
                await session.commit()
                
                logger.info(f"✅ Position #{position_id}: Trailing stop activated @ ${best_price}")
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='ACTIVATE_TRAILING',
                    new_trailing_active=True,
                    current_price=best_price,
                    reason=f"Trailing stop activated (distance: {trail_distance:.2%})",
                    triggered_by='SYSTEM'
                )
                
                return True
        except Exception as e:
            logger.error(f"❌ Error activating trailing stop: {e}")
            return False
    
    async def update_trailing_stop(
        self,
        position_id: int,
        new_best_price: float
    ) -> bool:
        """Оновлення trailing stop (нова найкраща ціна)"""
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    text("""
                        UPDATE positions
                        SET best_price = :new_best_price, updated_at = NOW()
                        WHERE id = :position_id
                    """),
                    {'position_id': position_id, 'new_best_price': new_best_price}
                )
                await session.commit()
                
                logger.info(f"✅ Position #{position_id}: New best price ${new_best_price}")
                
                # Логування в історію
                await self._add_history(
                    position_id=position_id,
                    event_type='UPDATE_TRAILING',
                    current_price=new_best_price,
                    reason=f"New best price: ${new_best_price}",
                    triggered_by='MONITOR'
                )
                
                return True
        except Exception as e:
            logger.error(f"❌ Error updating trailing stop: {e}")
            return False
    
    async def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Отримання відкритої позиції по символу"""
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT 
                            id, symbol, side, entry_price, quantity, leverage,
                            stop_loss_price, take_profit_price,
                            trailing_stop_active, trailing_stop_distance, best_price,
                            entry_time, ml_prediction, ml_confidence
                        FROM positions
                        WHERE symbol = :symbol AND status = 'open'
                        ORDER BY entry_time DESC
                        LIMIT 1
                    """),
                    {'symbol': symbol}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'symbol': row[1],
                        'side': row[2],
                        'entry_price': float(row[3]),
                        'quantity': float(row[4]),
                        'leverage': row[5],
                        'stop_loss_price': float(row[6]) if row[6] else None,
                        'take_profit_price': float(row[7]) if row[7] else None,
                        'trailing_stop_active': row[8],
                        'trailing_stop_distance': float(row[9]) if row[9] else None,
                        'best_price': float(row[10]) if row[10] else None,
                        'entry_time': row[11],
                        'ml_prediction': row[12],
                        'ml_confidence': float(row[13]) if row[13] else None
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Error getting position: {e}")
            return None
    
    async def get_all_open_positions(self) -> List[Dict]:
        """Отримання всіх відкритих позицій"""
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT 
                            id, symbol, side, entry_price, quantity, leverage,
                            stop_loss_price, take_profit_price,
                            trailing_stop_active, best_price, entry_time
                        FROM positions
                        WHERE status = 'open'
                        ORDER BY entry_time DESC
                    """)
                )
                rows = result.fetchall()
                
                positions = []
                for row in rows:
                    positions.append({
                        'id': row[0],
                        'symbol': row[1],
                        'side': row[2],
                        'entry_price': float(row[3]),
                        'quantity': float(row[4]),
                        'leverage': row[5],
                        'stop_loss_price': float(row[6]) if row[6] else None,
                        'take_profit_price': float(row[7]) if row[7] else None,
                        'trailing_stop_active': row[8],
                        'best_price': float(row[9]) if row[9] else None,
                        'entry_time': row[10]
                    })
                
                return positions
        except Exception as e:
            logger.error(f"❌ Error getting open positions: {e}")
            return []
    
    async def add_trade(
        self,
        position_id: int,
        symbol: str,
        side: str,
        trade_type: str,
        price: float,
        quantity: float,
        quote_quantity: float,
        realized_pnl: float = None,
        fee: float = 0.0,
        binance_trade_id: int = None,
        binance_order_id: int = None
    ) -> int:
        """Додавання трейду (заповнення ордера)"""
        try:
            async with self.db.async_session_factory() as session:
                result = await session.execute(
                    text("""
                        INSERT INTO trades (
                            position_id, symbol, side, trade_type,
                            price, quantity, quote_quantity,
                            realized_pnl, fee,
                            binance_trade_id, binance_order_id,
                            trade_time
                        ) VALUES (
                            :position_id, :symbol, :side, :trade_type,
                            :price, :quantity, :quote_quantity,
                            :realized_pnl, :fee,
                            :binance_trade_id, :binance_order_id,
                            NOW()
                        )
                        RETURNING id
                    """),
                    {
                        'position_id': position_id,
                        'symbol': symbol,
                        'side': side,
                        'trade_type': trade_type,
                        'price': price,
                        'quantity': quantity,
                        'quote_quantity': quote_quantity,
                        'realized_pnl': realized_pnl,
                        'fee': fee,
                        'binance_trade_id': binance_trade_id,
                        'binance_order_id': binance_order_id
                    }
                )
                trade_id = result.scalar_one()
                await session.commit()
                
                logger.info(f"✅ Trade added: #{trade_id} {symbol} {side} {trade_type}")
                return trade_id
        except Exception as e:
            logger.error(f"❌ Error adding trade: {e}")
            return None
    
    async def _add_history(
        self,
        position_id: int,
        event_type: str,
        old_status: str = None,
        new_status: str = None,
        old_stop_loss: float = None,
        new_stop_loss: float = None,
        old_take_profit: float = None,
        new_take_profit: float = None,
        old_trailing_active: bool = None,
        new_trailing_active: bool = None,
        current_price: float = None,
        unrealized_pnl: float = None,
        unrealized_pnl_pct: float = None,
        reason: str = None,
        triggered_by: str = 'SYSTEM'
    ):
        """Додавання запису в історію позиції"""
        try:
            async with self.db.async_session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO position_history (
                            position_id, event_type,
                            old_status, new_status,
                            old_stop_loss, new_stop_loss,
                            old_take_profit, new_take_profit,
                            old_trailing_active, new_trailing_active,
                            current_price, unrealized_pnl, unrealized_pnl_pct,
                            reason, triggered_by
                        ) VALUES (
                            :position_id, :event_type,
                            :old_status, :new_status,
                            :old_stop_loss, :new_stop_loss,
                            :old_take_profit, :new_take_profit,
                            :old_trailing_active, :new_trailing_active,
                            :current_price, :unrealized_pnl, :unrealized_pnl_pct,
                            :reason, :triggered_by
                        )
                    """),
                    {
                        'position_id': position_id,
                        'event_type': event_type,
                        'old_status': old_status,
                        'new_status': new_status,
                        'old_stop_loss': old_stop_loss,
                        'new_stop_loss': new_stop_loss,
                        'old_take_profit': old_take_profit,
                        'new_take_profit': new_take_profit,
                        'old_trailing_active': old_trailing_active,
                        'new_trailing_active': new_trailing_active,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': unrealized_pnl_pct,
                        'reason': reason,
                        'triggered_by': triggered_by
                    }
                )
                await session.commit()
        except Exception as e:
            logger.warning(f"⚠️ Error adding history: {e}")
