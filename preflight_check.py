#!/usr/bin/env python3
"""
🔍 ПРЕ-FLIGHT CHECK - Перевірка готовності до запуску
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("\n" + "="*80)
print("🔍 ПРЕ-FLIGHT CHECK - Перевірка готовності системи")
print("="*80 + "\n")

# 1. Перевірка .env
print("1️⃣ Перевірка .env файлу...")
load_dotenv()

api_key = os.getenv('FUTURES_API_KEY')
api_secret = os.getenv('FUTURES_API_SECRET')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat = os.getenv('TELEGRAM_CHAT_ID')

if api_key and api_secret:
    print(f"   ✅ FUTURES_API_KEY: {api_key[:10]}...")
    print(f"   ✅ FUTURES_API_SECRET: {api_secret[:10]}...")
else:
    print("   ❌ API ключі не знайдені!")
    sys.exit(1)

if telegram_token and telegram_chat:
    print(f"   ✅ TELEGRAM_BOT_TOKEN: {telegram_token[:10]}...")
    print(f"   ✅ TELEGRAM_CHAT_ID: {telegram_chat}")
else:
    print("   ⚠️ Telegram не налаштований (не критично)")

# 2. Перевірка моделей
print("\n2️⃣ Перевірка ML моделей...")
models_dir = Path('models')
required_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOTUSDT']
missing_models = []

for symbol in required_symbols:
    model_dir = models_dir / f'simple_trend_{symbol}'
    model_file = model_dir / f'model_{symbol}_4h.pkl'
    
    if model_file.exists():
        print(f"   ✅ {symbol} - готова")
    else:
        print(f"   ❌ {symbol} - ВІДСУТНЯ")
        missing_models.append(symbol)

if missing_models:
    print(f"\n   ⚠️ УВАГА: Відсутні моделі для {len(missing_models)} символів")
    print(f"   Запустіть: python master_control.py train --symbols {' '.join(missing_models)}")
    sys.exit(1)

# 3. Перевірка залежностей
print("\n3️⃣ Перевірка Python пакетів...")
required_packages = [
    'binance',
    'pandas',
    'numpy',
    'joblib',
    'sklearn',
    'telegram',
    'dotenv',
    'tabulate',
    'sqlalchemy',
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package if package != 'sklearn' else 'sklearn')
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package}")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️ УВАГА: Відсутні пакети: {', '.join(missing_packages)}")
    print(f"   Запустіть: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# 4. Перевірка Rust індикаторів
print("\n4️⃣ Перевірка Rust індикаторів...")
try:
    from training.rust_features import RustFeatureEngineer
    engineer = RustFeatureEngineer()
    print("   ✅ Rust індикатори доступні")
except Exception as e:
    print(f"   ⚠️ Rust індикатори не доступні: {e}")
    print("   (Не критично, але може уповільнити роботу)")

# 5. Тест підключення до Binance
print("\n5️⃣ Тест підключення до Binance Testnet...")
try:
    from binance.client import Client
    client = Client(api_key, api_secret, testnet=True)
    account = client.futures_account()
    balance = float(account['totalWalletBalance'])
    print(f"   ✅ Підключення успішне")
    print(f"   💰 Баланс: ${balance:.2f} USDT")
except Exception as e:
    print(f"   ❌ Помилка підключення: {e}")
    sys.exit(1)

# 6. Перевірка відкритих позицій
print("\n6️⃣ Перевірка поточних позицій...")
try:
    positions = client.futures_position_information()
    open_positions = [p for p in positions if abs(float(p['positionAmt'])) > 0.0001]
    
    if open_positions:
        print(f"   ⚠️ Вже є {len(open_positions)} відкритих позицій:")
        total_pnl = 0
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            pnl = float(pos['unRealizedProfit'])
            side = 'LONG' if amt > 0 else 'SHORT'
            print(f"      • {symbol} {side}: ${pnl:+.2f}")
            total_pnl += pnl
        print(f"   💰 Загальний PnL: ${total_pnl:+.2f}")
    else:
        print("   ✅ Немає відкритих позицій")
except Exception as e:
    print(f"   ⚠️ Помилка перевірки позицій: {e}")

# 7. Перевірка логів
print("\n7️⃣ Перевірка директорії логів...")
logs_dir = Path('logs')
if not logs_dir.exists():
    logs_dir.mkdir()
    print("   ✅ Директорія logs створена")
else:
    log_files = list(logs_dir.glob('*.log'))
    print(f"   ✅ Директорія logs існує ({len(log_files)} файлів)")

# Підсумок
print("\n" + "="*80)
print("✅ ВСІ ПЕРЕВІРКИ ПРОЙДЕНО УСПІШНО!")
print("="*80)
print("\n🚀 Готово до запуску нічної торгівлі:")
print("   python night_trading.py")
print("\n💡 Додатково:")
print("   • Перевірити позиції: python check_orders.py")
print("   • Моніторинг: python master_control.py monitor")
print("   • Зупинити: Ctrl+C")
print("\n⚠️ УВАГА: Бот буде торгувати на Testnet з реальними ордерами!")
print("   Переконайтесь, що ви готові до цього.\n")
