#!/usr/bin/env python3
"""
Закриття всіх відкритих позицій
"""
from binance.client import Client
from config import BINANCE_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def close_all_positions():
    """Закрити всі відкриті позиції"""
    client = Client(
        BINANCE_CONFIG['api_key'],
        BINANCE_CONFIG['api_secret'],
        testnet=BINANCE_CONFIG['testnet']
    )
    
    try:
        # Отримати всі позиції
        positions = client.futures_position_information()
        
        closed_count = 0
        total_pnl = 0.0
        
        logger.info("="*80)
        logger.info("🔍 Пошук відкритих позицій...")
        logger.info("="*80)
        
        for pos in positions:
            position_amt = float(pos['positionAmt'])
            if position_amt == 0:
                continue
            
            symbol = pos['symbol']
            entry_price = float(pos['entryPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            side = 'LONG' if position_amt > 0 else 'SHORT'
            
            logger.info(f"\n📊 {symbol}: {side} {abs(position_amt)} @ ${entry_price:.2f}")
            logger.info(f"   💰 P&L: ${unrealized_pnl:.2f}")
            
            # Скасувати всі ордери
            try:
                client.futures_cancel_all_open_orders(symbol=symbol)
                logger.info(f"   ✅ Скасовано всі ордери для {symbol}")
            except Exception as e:
                logger.warning(f"   ⚠️ Помилка скасування ордерів: {e}")
            
            # Закрити позицію (ринковий ордер у протилежний бік)
            try:
                side_to_close = 'SELL' if position_amt > 0 else 'BUY'
                
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side_to_close,
                    type='MARKET',
                    quantity=abs(position_amt),
                    reduceOnly=True
                )
                
                logger.info(f"   ✅ Закрито {symbol}: {side} ${unrealized_pnl:+.2f}")
                closed_count += 1
                total_pnl += unrealized_pnl
                
            except Exception as e:
                logger.error(f"   ❌ Помилка закриття {symbol}: {e}")
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ ЗАКРИТО: {closed_count} позицій")
        logger.info(f"💰 ЗАГАЛЬНИЙ P&L: ${total_pnl:+.2f}")
        logger.info("="*80)
        
        if closed_count == 0:
            logger.info("ℹ️ Немає відкритих позицій")
        else:
            import time
            logger.info("\n⏳ Почекайте 5 секунд, щоб бот побачив зміни...")
            time.sleep(5)
            logger.info("✅ Готово! Бот може відкривати нові позиції з правильними SL/TP")
        
    except Exception as e:
        logger.error(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    close_all_positions()
