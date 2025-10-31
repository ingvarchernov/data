#!/usr/bin/env python3
"""
Скрипт для перевірки відкритих ордерів та позицій на Binance Futures Testnet
"""
import asyncio
import logging
import os
from binance.client import Client
from datetime import datetime
from tabulate import tabulate
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderChecker:
    def __init__(self, testnet: bool = True):
        """Ініціалізація клієнта Binance"""
        if testnet:
            # Спробуємо отримати ключі з змінних середовища
            api_key = (
                os.getenv('FUTURES_API_KEY') or
                os.getenv('BINANCE_TEST_API_KEY') or
                input("Введіть Binance Testnet API Key: ").strip()
            )
            api_secret = (
                os.getenv('FUTURES_API_SECRET') or
                os.getenv('BINANCE_TEST_API_SECRET') or
                input("Введіть Binance Testnet API Secret: ").strip()
            )

            if not api_key or not api_secret:
                raise ValueError("❌ API ключі не знайдені. Створіть .env файл або введіть їх вручну.")

            logger.info("🔧 Підключення до Binance Futures TESTNET")
        else:
            raise NotImplementedError("Real trading не реалізовано")

        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet

    async def check_account_balance(self):
        """Перевірка балансу рахунку"""
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.futures_account()
            )

            balance = float(account['totalWalletBalance'])
            available = float(account['availableBalance'])
            unrealized_pnl = float(account['totalUnrealizedProfit'])

            print("\n" + "="*50)
            print("💰 БАЛАНС РАХУНКУ")
            print("="*50)
            print(f"💼 Загальний баланс:     ${balance:,.2f}")
            print(f"💵 Доступний баланс:     ${available:,.2f}")
            print(f"📊 Нереалізований PnL:   ${unrealized_pnl:,.2f}")

            return account

        except Exception as e:
            logger.error(f"❌ Помилка отримання балансу: {e}")
            return None

    async def check_open_orders(self):
        """Перевірка відкритих ордерів"""
        try:
            orders = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.futures_get_open_orders()
            )

            print("\n" + "="*80)
            print("📋 ВІДКРИТІ ОРДЕРИ")
            print("="*80)

            if not orders:
                print("✅ Немає відкритих ордерів")
                return []

            # Підготовка даних для таблиці
            table_data = []
            for order in orders:
                table_data.append([
                    order['symbol'],
                    order['side'],
                    order['type'],
                    f"{float(order['origQty']):.4f}",
                    f"${float(order['price']):,.2f}" if order['price'] != '0' else "MARKET",
                    order['status'],
                    datetime.fromtimestamp(order['time']/1000).strftime('%H:%M:%S')
                ])

            headers = ['Symbol', 'Side', 'Type', 'Quantity', 'Price', 'Status', 'Time']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))

            return orders

        except Exception as e:
            logger.error(f"❌ Помилка отримання ордерів: {e}")
            return []

    async def check_open_positions(self):
        """Перевірка відкритих позицій"""
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.futures_position_information()
            )

            # Фільтруємо тільки відкриті позиції
            open_positions = [
                pos for pos in positions
                if abs(float(pos['positionAmt'])) > 0.0001
            ]

            print("\n" + "="*100)
            print("📈 ВІДКРИТІ ПОЗИЦІЇ")
            print("="*100)

            if not open_positions:
                print("✅ Немає відкритих позицій")
                return []

            # Підготовка даних для таблиці
            table_data = []
            for pos in open_positions:
                amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                mark_price = float(pos['markPrice'])
                unrealized_pnl = float(pos['unRealizedProfit'])
                
                # Розраховуємо leverage з notional та margin
                notional = abs(float(pos['notional']))  # Вартість позиції
                initial_margin = float(pos['positionInitialMargin'])  # Використаний margin
                leverage = round(notional / initial_margin) if initial_margin > 0 else 1
                
                # ПРАВИЛЬНИЙ розрахунок PnL% з врахуванням leverage:
                # PnL% = (unrealized_pnl / initial_margin) * 100
                pnl_percent = (unrealized_pnl / initial_margin) * 100 if initial_margin > 0 else 0

                table_data.append([
                    pos['symbol'],
                    'LONG' if amt > 0 else 'SHORT',
                    f"{abs(amt):.4f}",
                    f"${entry_price:,.4f}",
                    f"${mark_price:,.4f}",
                    f"${unrealized_pnl:,.2f}",
                    f"{pnl_percent:+.2f}%",
                    f"{leverage}x"
                ])

            headers = ['Symbol', 'Side', 'Size', 'Entry', 'Mark', 'PnL ($)', 'PnL (%)', 'Leverage']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))

            return open_positions

        except Exception as e:
            logger.error(f"❌ Помилка отримання позицій: {e}")
            return []

    async def check_recent_trades(self, limit: int = 10):
        """Перевірка останніх угод"""
        try:
            # Отримуємо список відкритих позицій для фільтрації
            positions = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.futures_position_information()
            )

            active_symbols = [
                pos['symbol'] for pos in positions
                if abs(float(pos['positionAmt'])) > 0.0001
            ]

            if not active_symbols:
                print("\n✅ Немає активних символів для перевірки угод")
                return

            print("\n" + "="*100)
            print("📊 ОСТАННІ УГОДИ")
            print("="*100)

            all_trades = []

            for symbol in active_symbols[:5]:  # Обмежуємо до 5 символів
                try:
                    trades = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=symbol: self.client.futures_account_trades(symbol=s, limit=5)
                    )

                    for trade in trades[-3:]:  # Останні 3 угоди
                        all_trades.append([
                            trade['symbol'],
                            'BUY' if trade['buyer'] else 'SELL',
                            f"{float(trade['qty']):.4f}",
                            f"${float(trade['price']):,.4f}",
                            f"${float(trade['realizedPnl']):,.2f}",
                            datetime.fromtimestamp(trade['time']/1000).strftime('%m/%d %H:%M')
                        ])

                except Exception as trade_err:
                    logger.warning(f"⚠️ Помилка угод для {symbol}: {trade_err}")

            if all_trades:
                # Сортуємо за часом
                all_trades.sort(key=lambda x: x[5], reverse=True)
                headers = ['Symbol', 'Side', 'Qty', 'Price', 'PnL', 'Time']
                print(tabulate(all_trades[:limit], headers=headers, tablefmt='grid'))
            else:
                print("❌ Не вдалося отримати дані про угоди")

        except Exception as e:
            logger.error(f"❌ Помилка отримання угод: {e}")

    async def run_full_check(self):
        """Повна перевірка рахунку"""
        print("🚀 Запуск перевірки Binance Futures рахунку...")

        # Перевірка балансу
        await self.check_account_balance()

        # Перевірка ордерів
        await self.check_open_orders()

        # Перевірка позицій
        await self.check_open_positions()

        # Перевірка угод
        await self.check_recent_trades()

        print("\n✅ Перевірка завершена!")

async def main():
    """Головна функція"""
    checker = OrderChecker(testnet=True)
    await checker.run_full_check()

if __name__ == "__main__":
    asyncio.run(main())