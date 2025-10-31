"""
Analytics - аналітика та моніторинг системи
"""
import asyncio
from datetime import datetime
import logging

from telegram_bot import telegram_notifier

logger = logging.getLogger(__name__)


class TradingSession:
    """Статистика торгової сесії"""
    def __init__(self):
        self.start_time = datetime.now()
        self.iterations = 0
        self.signals_generated = 0
        self.positions_opened = 0
        self.positions_closed = 0
        self.balance_history = []
        self.peak_balance = 0.0
        
    def duration(self):
        """Тривалість сесії"""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}год {minutes}хв {seconds}сек"


async def get_analytics(bot, session: TradingSession, iteration: int):
    """
    Отримання аналітики на поточній ітерації
    
    Args:
        bot: TradingBot instance
        session: TradingSession instance  
        iteration: Номер ітерації
    """
    try:
        leverage = bot.leverage
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 АНАЛІТИКА - Ітерація #{iteration}")
        logger.info(f"{'='*80}")
        
        # Отримуємо дані рахунку
        account = await asyncio.get_event_loop().run_in_executor(
            None, bot.client.futures_account
        )
        
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        unrealized_pnl = float(account['totalUnrealizedProfit'])
        margin_used = balance - available
        margin_ratio = (margin_used / balance * 100) if balance > 0 else 0
        
        # Оновлення історії балансу
        session.balance_history.append(balance)
        if balance > session.peak_balance:
            session.peak_balance = balance
        
        # Drawdown
        drawdown = 0.0
        if session.peak_balance > 0:
            drawdown = ((session.peak_balance - balance) / session.peak_balance) * 100
        
        logger.info(f"\n💰 РАХУНОК:")
        logger.info(f"   Баланс: ${balance:.2f}")
        logger.info(f"   Доступно: ${available:.2f}")
        logger.info(f"   Unrealized PnL: ${unrealized_pnl:.2f}")
        logger.info(f"   Маржа використана: {margin_ratio:.1f}%")
        logger.info(f"   Peak баланс: ${session.peak_balance:.2f}")
        logger.info(f"   Drawdown: {drawdown:.2f}%")
        
        # Позиції
        positions = await asyncio.get_event_loop().run_in_executor(
            None, bot.client.futures_position_information
        )
        
        active_positions = []
        total_pos_pnl = 0.0
        
        for pos in positions:
            amount = float(pos['positionAmt'])
            if amount == 0:
                continue
            
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized = float(pos['unRealizedProfit'])
            
            # PnL%
            notional = abs(amount * mark_price)
            initial_margin = notional / leverage
            pnl_pct = (unrealized / initial_margin * 100) if initial_margin > 0 else 0
            
            side = 'LONG' if amount > 0 else 'SHORT'
            
            active_positions.append({
                'symbol': pos['symbol'],
                'side': side,
                'size': abs(amount),
                'entry': entry_price,
                'mark': mark_price,
                'pnl': unrealized,
                'pnl_pct': pnl_pct,
            })
            total_pos_pnl += unrealized
        
        logger.info(f"\n📈 ПОЗИЦІЇ: {len(active_positions)}")
        if active_positions:
            for p in sorted(active_positions, key=lambda x: abs(x['pnl']), reverse=True):
                logger.info(
                    f"   {p['symbol']:<10} {p['side']:<5} "
                    f"${p['pnl']:>7.2f} ({p['pnl_pct']:>+6.2f}%) "
                    f"@ ${p['mark']:.8g}"
                )
            logger.info(f"   {'─'*50}")
            logger.info(f"   {'ВСЬОГО':<16} ${total_pos_pnl:>7.2f}")
        
        # Ордери
        orders = await asyncio.get_event_loop().run_in_executor(
            None, bot.client.futures_get_open_orders
        )
        
        sl_count = sum(1 for o in orders if o['type'] == 'STOP_MARKET')
        tp_count = sum(1 for o in orders if o['type'] == 'TAKE_PROFIT_MARKET')
        
        logger.info(f"\n📋 ОРДЕРИ: {len(orders)}")
        if orders:
            logger.info(f"   Stop-Loss: {sl_count}")
            logger.info(f"   Take-Profit: {tp_count}")
        
        # Статистика сесії
        logger.info(f"\n⏱️  СЕСІЯ:")
        logger.info(f"   Тривалість: {session.duration()}")
        logger.info(f"   Ітерацій: {session.iterations}")
        logger.info(f"   Сигналів: {session.signals_generated}")
        logger.info(f"   Позицій відкрито: {session.positions_opened}")
        logger.info(f"   Позицій закрито: {session.positions_closed}")
        
        # Health check
        issues = []
        warnings = []
        
        if balance < 1000:
            issues.append("⚠️ Низький баланс (< $1000)")
        
        if margin_ratio > 80:
            issues.append(f"🚨 Висока маржа ({margin_ratio:.1f}%)")
        elif margin_ratio > 50:
            warnings.append(f"⚠️ Маржа {margin_ratio:.1f}%")
        
        if drawdown > 20:
            issues.append(f"🚨 Великий drawdown ({drawdown:.1f}%)")
        elif drawdown > 10:
            warnings.append(f"⚠️ Drawdown {drawdown:.1f}%")
        
        # Перевірка позицій без захисту
        for pos in active_positions:
            symbol = pos['symbol']
            symbol_orders = [o for o in orders if o['symbol'] == symbol]
            has_sl = any(o['type'] == 'STOP_MARKET' for o in symbol_orders)
            has_tp = any(o['type'] == 'TAKE_PROFIT_MARKET' for o in symbol_orders)
            if not has_sl or not has_tp:
                warnings.append(f"⚠️ {symbol} без захисту (SL:{has_sl}, TP:{has_tp})")
        
        # Великі втрати
        for pos in active_positions:
            if pos['pnl_pct'] < -50:
                issues.append(f"🚨 {pos['symbol']}: втрати {pos['pnl_pct']:.1f}%")
            elif pos['pnl_pct'] < -30:
                issues.append(f"⚠️ {pos['symbol']}: втрати {pos['pnl_pct']:.1f}%")
        
        status = '✅ OK' if not issues else ('⚠️ УВАГА' if not any('🚨' in i for i in issues) else '🚨 КРИТИЧНО')
        
        logger.info(f"\n🏥 СТАН СИСТЕМИ: {status}")
        for warning in warnings:
            logger.warning(f"   {warning}")
        for issue in issues:
            logger.error(f"   {issue}")
        
        logger.info(f"{'='*80}\n")
        
        # Telegram (кожні 10 ітерацій або при проблемах)
        if iteration % 10 == 0 or issues:
            message = (
                f"📊 ЗВІТ #{iteration}\n"
                f"{'━'*30}\n"
                f"💰 Баланс: ${balance:.2f}\n"
                f"📊 PnL: ${unrealized_pnl:.2f}\n"
                f"📈 Позицій: {len(active_positions)}\n"
                f"⏱️ Сесія: {session.duration()}\n"
                f"\n🏥 Стан: {status}\n"
            )
            
            if issues:
                message += "\n🚨 ПРОБЛЕМИ:\n" + "\n".join(issues) + "\n"
            if warnings:
                message += "\n⚠️ ПОПЕРЕДЖЕННЯ:\n" + "\n".join(warnings[:3]) + "\n"
            
            if active_positions:
                # ТОП-3 НАЙКРАЩІ (найбільший прибуток)
                best_positions = sorted(active_positions, key=lambda x: x['pnl'], reverse=True)[:3]
                if any(p['pnl'] > 0 for p in best_positions):
                    message += f"\n� ТОП-3 НАЙКРАЩІ:\n"
                    for p in best_positions:
                        if p['pnl'] > 0:
                            message += f"{p['symbol']} {p['side']}: ${p['pnl']:.2f} ({p['pnl_pct']:+.1f}%)\n"
                
                # ТОП-3 НАЙГІРШІ (найбільший збиток)
                worst_positions = sorted(active_positions, key=lambda x: x['pnl'])[:3]
                if any(p['pnl'] < 0 for p in worst_positions):
                    message += f"\n🔴 ТОП-3 НАЙГІРШІ:\n"
                    for p in worst_positions:
                        if p['pnl'] < 0:
                            message += f"{p['symbol']} {p['side']}: ${p['pnl']:.2f} ({p['pnl_pct']:+.1f}%)\n"
            
            await telegram_notifier.send_message(message)
        
        return {
            'balance': balance,
            'positions_count': len(active_positions),
            'total_pnl': total_pos_pnl,
            'healthy': len(issues) == 0,
        }
        
    except Exception as e:
        logger.error(f"❌ Помилка аналітики: {e}")
        return None
