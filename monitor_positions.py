#!/usr/bin/env python3
"""
Простий скрипт для моніторингу позицій без ML моделей
"""
import asyncio
import os
from dotenv import load_dotenv
from binance.client import Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def monitor_positions():
    """Моніторинг позицій"""
    # API ключі
    api_key = os.getenv('FUTURES_API_KEY')
    api_secret = os.getenv('FUTURES_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Встановіть FUTURES_API_KEY та FUTURES_API_SECRET в .env файлі")
        return
    
    # Binance client
    client = Client(api_key, api_secret, testnet=True)
    logger.info("✅ Підключено до Binance Testnet")
    
    # Символи для моніторингу
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'VETUSDT']
    
    print("\n" + "="*80)
    print("📊 МОНІТОРИНГ ПОЗИЦІЙ")
    print("="*80)
    
    total_pnl = 0.0
    position_count = 0
    
    for symbol in symbols:
        try:
            # Отримуємо інформацію про позицію
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=symbol: client.futures_position_information(symbol=s)
            )
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    amt = float(pos['positionAmt'])
                    if abs(amt) > 0.0001:  # Позиція відкрита
                        entry_price = float(pos['entryPrice'])
                        mark_price = float(pos['markPrice'])
                        unrealized_pnl = float(pos['unRealizedProfit'])
                        side = 'LONG' if amt > 0 else 'SHORT'
                        
                        pnl_pct = (unrealized_pnl / (abs(amt) * entry_price)) * 100 if entry_price > 0 else 0
                        
                        emoji_side = "📈" if side == 'LONG' else "📉"
                        emoji_pnl = "💰" if unrealized_pnl > 0 else "📉" if unrealized_pnl < 0 else "⚪"
                        
                        print(f"\n{emoji_side} {symbol}:")
                        print(f"   Side: {side}")
                        print(f"   Size: {abs(amt):.4f}")
                        print(f"   Entry: ${entry_price:,.4f}")
                        print(f"   Mark:  ${mark_price:,.4f}")
                        print(f"   {emoji_pnl} PnL: ${unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                        
                        total_pnl += unrealized_pnl
                        position_count += 1
                        
        except Exception as e:
            logger.error(f"❌ Помилка для {symbol}: {e}")
    
    print("\n" + "="*80)
    if position_count > 0:
        emoji_total = "💰" if total_pnl > 0 else "📉" if total_pnl < 0 else "⚪"
        print(f"📊 Всього позицій: {position_count}")
        print(f"{emoji_total} Загальний PnL: ${total_pnl:+.2f}")
    else:
        print("✅ Немає відкритих позицій")
    print("="*80)
    
    # Отримуємо баланс
    try:
        account = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.futures_account()
        )
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        print(f"\n💼 Баланс: ${balance:,.2f}")
        print(f"💵 Доступно: ${available:,.2f}")
    except Exception as e:
        logger.error(f"❌ Помилка отримання балансу: {e}")

async def main():
    """Головна функція"""
    try:
        await monitor_positions()
    except Exception as e:
        logger.error(f"❌ Помилка: {e}")

if __name__ == "__main__":
    asyncio.run(main())