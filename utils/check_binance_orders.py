"""
Перевірка відкритих ордерів на Binance
"""
import os
import sys
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# Testnet API
api_key = os.getenv('FUTURES_API_KEY')
api_secret = os.getenv('FUTURES_API_SECRET')

client = Client(api_key, api_secret, testnet=True)

print("=" * 60)
print("🔍 CHECKING BINANCE ORDERS")
print("=" * 60)
print()

# Open positions
print("1️⃣ Open Positions:")
positions = client.futures_position_information()
open_pos = [p for p in positions if abs(float(p['positionAmt'])) > 0.0001]

if not open_pos:
    print("   ❌ No open positions")
else:
    try:
        for pos in open_pos:
            symbol = pos['symbol']
            orders = client.futures_get_open_orders(symbol=symbol)
            
            if not orders:
                print(f"   ❌ {symbol}: NO ORDERS (TP/SL відсутні!)")
            else:
                print(f"   ✅ {symbol}: {len(orders)} orders")
                for order in orders:
                    otype = order['type']
                    side = order['side']
                    price = float(order['stopPrice']) if 'stopPrice' in order else float(order['price'])
                    qty = float(order['origQty'])
                    
                    print(f"      • {otype} {side}: {qty} @ ${price:.4f}")
                    print(f"        Order ID: {order['orderId']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
            
print(f"=" * 60)
