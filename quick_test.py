#!/usr/bin/env python3
"""
Швидке тестування основних функцій системи
"""
import asyncio
import os
from datetime import datetime

from binance_data_loader import BinanceDataLoader
from error_handling import TradingLogger
from strategies.scalping import ScalpingStrategy
from trading.live_trading.binance_testnet import BinanceTestnetTrader


async def quick_system_test():
    """Швидке тестування основних компонентів"""
    print("🚀 ШВИДКЕ ТЕСТУВАННЯ СИСТЕМИ")
    print("=" * 40)

    # 1. Тест логування
    print("\n1️⃣ Тестування системи логування...")
    try:
        logger = TradingLogger("quick_test")
        logger.log_trade({"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.001})
        print("✅ Логування працює")
    except Exception as e:
        print(f"❌ Помилка логування: {e}")

    # 2. Тест завантаження даних
    print("\n2️⃣ Тестування завантаження даних...")
    try:
        loader = BinanceDataLoader()
        data = loader.get_historical_klines('BTCUSDT', '1h', limit=10)
        if not data.empty:
            print(f"✅ Завантажено {len(data)} свічок BTCUSDT")
            print(f"   Остання ціна: {data['close'].iloc[-1]:.2f} USDT")
        else:
            print("⚠️ Немає даних (можливо проблеми з API)")
    except Exception as e:
        print(f"❌ Помилка завантаження даних: {e}")

    # 3. Тест стратегії
    print("\n3️⃣ Тестування торгової стратегії...")
    try:
        strategy = ScalpingStrategy(['BTCUSDT'])
        print(f"✅ Стратегія {strategy.__class__.__name__} ініціалізована")
        print(f"   Символи: {strategy.symbols}")
    except Exception as e:
        print(f"❌ Помилка стратегії: {e}")

    # 4. Тест Binance Testnet (якщо є ключі)
    print("\n4️⃣ Тестування Binance Testnet...")
    try:
        if os.getenv('BINANCE_TEST_API_KEY') and os.getenv('BINANCE_TEST_API_SECRET'):
            trader = BinanceTestnetTrader()
            balance = trader.get_account_balance()
            print("✅ Binance Testnet підключено")
            print(f"   Загальна кількість активів: {len(balance)}")

            # Показати основні активи
            for asset, amount in list(balance.items())[:5]:
                print(f"   {asset}: {amount}")

            # Отримати поточну ціну
            try:
                btc_price = trader.get_current_price('BTCUSDT')
                print(f"   Поточна ціна BTCUSDT: {btc_price:.2f} USDT")
            except Exception as e:
                print(f"   Не вдалося отримати ціну BTC: {e}")

        else:
            print("⚠️ Немає API ключів Binance Testnet (BINANCE_TEST_API_KEY, BINANCE_TEST_API_SECRET)")

    except Exception as e:
        print(f"❌ Помилка Binance Testnet: {e}")

    print("\n" + "=" * 40)
    print("🎯 ШВИДКЕ ТЕСТУВАННЯ ЗАВЕРШЕНО")


async def test_small_trade():
    """Тестування маленької угоди"""
    print("\n💰 ТЕСТ МАЛЕНЬКОЇ УГОДИ")
    print("=" * 30)

    try:
        # Перевірка наявності ключів
        if not (os.getenv('BINANCE_TEST_API_KEY') and os.getenv('BINANCE_TEST_API_SECRET')):
            print("❌ Немає API ключів для тестування угод")
            return

        trader = BinanceTestnetTrader()
        balance = trader.get_account_balance()

        print(f"Баланс до угоди: {balance.get('USDT', 0):.2f} USDT")

        # Отримати поточну ціну
        btc_price = trader.get_current_price('BTCUSDT')
        print(f"Поточна ціна BTC: {btc_price:.2f} USDT")

        # Розрахувати маленьку кількість (еквівалент 1 USDT)
        quantity = 1.0 / btc_price  # ~0.000025 BTC для ціни 40000
        quantity = round(quantity, 6)  # Округлення для BTC

        print(f"Тестова кількість: {quantity} BTC (~{quantity * btc_price:.2f} USDT)")

        # Підтвердження
        confirm = input("Виконати тестову угоду? (yes/no): ")
        if confirm.lower() not in ['yes', 'y', 'так']:
            print("❌ Угоду скасовано")
            return

        # BUY угода
        print("🔄 Виконання BUY угоди...")
        buy_result = trader.place_market_order('BTCUSDT', 'BUY', quantity)

        if buy_result:
            print("✅ BUY угода виконана:")
            print(f"   Ціна: {buy_result.get('price', 'N/A')}")
            print(f"   Кількість: {buy_result.get('quantity', 'N/A')}")
            print(f"   Сума: {buy_result.get('cummulativeQuoteQty', 'N/A')}")

            # Очікування 30 секунд
            print("⏳ Очікування 30 секунд...")
            await asyncio.sleep(30)

            # SELL угода
            print("🔄 Виконання SELL угоди...")
            sell_result = trader.place_market_order('BTCUSDT', 'SELL', quantity)

            if sell_result:
                print("✅ SELL угода виконана:")
                print(f"   Ціна: {sell_result.get('price', 'N/A')}")
                print(f"   Кількість: {sell_result.get('quantity', 'N/A')}")

                # Розрахунок P&L
                buy_price = float(buy_result.get('price', 0))
                sell_price = float(sell_result.get('price', 0))
                pnl = (sell_price - buy_price) * quantity

                print(f"💰 P&L: {pnl:.4f} USDT ({(pnl/buy_price*100):.4f}%)")

                # Баланс після
                final_balance = trader.get_account_balance()
                print(f"Баланс після угоди: {final_balance.get('USDT', 0):.2f} USDT")

            else:
                print("❌ Помилка SELL угоди")
        else:
            print("❌ Помилка BUY угоди")

    except Exception as e:
        print(f"❌ Помилка тестування угоди: {e}")


async def main():
    """Головна функція"""
    print("Виберіть тип тестування:")
    print("1. Швидке тестування системи")
    print("2. Тест маленької угоди")
    print("3. Повне тестування (real_trading_test.py)")

    choice = input("Ваш вибір (1-3): ").strip()

    if choice == '1':
        await quick_system_test()
    elif choice == '2':
        await test_small_trade()
    elif choice == '3':
        print("Запуск повного тестування...")
        os.system("python real_trading_test.py")
    else:
        print("❌ Неправильний вибір")


if __name__ == "__main__":
    asyncio.run(main())