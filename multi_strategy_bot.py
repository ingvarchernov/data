#!/usr/bin/env python3
"""
Multi-Strategy Trading Bot
Запускає кілька стратегій одночасно на різних таймфреймах
"""
import asyncio
import argparse
import logging
import os
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv

from strategies.trend_strategy_4h import TrendStrategy4h
from strategy_manager import StrategyManager
from telegram_bot import TelegramNotifier

# Завантажуємо .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Telegram notifier
telegram_notifier = TelegramNotifier()

# Символи за accuracy
SYMBOLS_70_PLUS = ['BTCUSDT', 'TRXUSDT', 'LTCUSDT']  # ≥70%
SYMBOLS_60_70 = ['XRPUSDT', 'BNBUSDT', 'ALGOUSDT', 'ETHUSDT', 'UNIUSDT', 'XLMUSDT']  # 60-70%


class MultiStrategyBot:
    """Мультистратегійний торговий бот"""
    
    def __init__(
        self,
        testnet: bool = True,
        enable_trading: bool = False,
        use_all_symbols: bool = False
    ):
        # Binance client - використовуємо ключі з .env
        if testnet:
            api_key = os.getenv('FUTURES_API_KEY')
            api_secret = os.getenv('FUTURES_API_SECRET')
            
            if not api_key or not api_secret:
                logger.warning("⚠️ API ключі не знайдені в .env, використовуємо старі")
                api_key = "9be7e1fae31b26ee6d3be23e1c3e4c8eca0d7a265e37fef5f91259e3a4cf9286"
                api_secret = "ba1b2cfce24e65f6f374e1e78c7ea5e803ec1cab06d89e4b3e7b5ba46e4b20a6"
            
            logger.info("✅ Binance client (TESTNET)")
        else:
            raise NotImplementedError("Real trading не реалізовано")
        
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        self.enable_trading = enable_trading
        
        # Strategy Manager
        self.strategy_manager = StrategyManager(self.client)
        
        # Вибір символів
        if use_all_symbols:
            symbols_4h = SYMBOLS_70_PLUS + SYMBOLS_60_70
            logger.info(f"📊 Використовуємо ВСІ символи: {len(symbols_4h)}")
        else:
            symbols_4h = SYMBOLS_70_PLUS
            logger.info(f"📊 Використовуємо ТОП символи (≥70%): {len(symbols_4h)}")
        
        # Додаємо стратегії
        self._setup_strategies(symbols_4h)
        
        if not enable_trading:
            logger.info("ℹ️ Demo режим (угоди не виконуються)")
        else:
            logger.warning("⚠️ РЕАЛЬНІ УГОДИ УВІМКНЕНІ!")
    
    def _setup_strategies(self, symbols_4h: list):
        """Налаштування стратегій"""
        
        # 4h Trend Strategy
        trend_4h = TrendStrategy4h(
            symbols=symbols_4h,
            testnet=self.testnet,
            min_confidence=0.70,
            risk_per_trade=0.01
        )
        self.strategy_manager.add_strategy(trend_4h)
        
        # TODO: Додати 1h Swing Strategy коли натренуємо моделі
        # swing_1h = SwingStrategy1h(...)
        # self.strategy_manager.add_strategy(swing_1h)
    
    async def get_balance(self) -> float:
        """Баланс USDT"""
        try:
            loop = asyncio.get_event_loop()
            account = await loop.run_in_executor(
                None,
                self.client.futures_account
            )
            
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    balance = float(asset['walletBalance'])
                    logger.info(f"💰 Баланс: ${balance:.2f} USDT")
                    return balance
            
            return 0.0
        except Exception as e:
            logger.error(f"❌ Помилка балансу: {e}")
            return 0.0
    
    async def execute_signal(self, signal):
        """Виконання торгового сигналу"""
        try:
            logger.info(f"\n🎯 СИГНАЛ: {signal.action} {signal.symbol}")
            logger.info(f"   Стратегія: {signal.strategy}")
            logger.info(f"   Впевненість: {signal.confidence:.2%}")
            logger.info(f"   Ціна: ${signal.price:.2f}")
            
            # Telegram notification
            await telegram_notifier.send_message(
                f"🎯 {signal.action} SIGNAL\n"
                f"Symbol: {signal.symbol}\n"
                f"Strategy: {signal.strategy}\n"
                f"Confidence: {signal.confidence:.2%}\n"
                f"Price: ${signal.price:.2f}\n"
                f"{'🔴 REAL' if self.enable_trading else '🟡 DEMO'}"
            )
            
            if not self.enable_trading:
                logger.info("   ⚠️ DEMO MODE - угода НЕ виконана")
                return
            
            # TODO: Виконання реальних ордерів
            # if signal.action == 'BUY':
            #     await self.open_position(signal)
            # elif signal.action == 'SELL' or signal.action == 'CLOSE':
            #     await self.close_position(signal)
            
        except Exception as e:
            logger.error(f"❌ Помилка виконання сигналу: {e}", exc_info=True)
    
    async def run(self):
        """Головний цикл"""
        logger.info("="*70)
        logger.info("🚀 MULTI-STRATEGY TRADING BOT")
        logger.info("="*70)
        
        # Ініціалізація
        await self.strategy_manager.initialize()
        
        # Telegram старт
        await telegram_notifier.send_message(
            f"🚀 Multi-Strategy Bot запущено\n"
            f"Стратегій: {len(self.strategy_manager.strategies)}\n"
            f"Режим: {'🔴 РЕАЛЬНІ УГОДИ' if self.enable_trading else '🟡 DEMO'}"
        )
        
        # Баланс
        await self.get_balance()
        
        # Статистика стратегій
        self.strategy_manager.print_stats()
        
        # Мінімальний інтервал
        min_interval = self.strategy_manager.get_min_interval()
        logger.info(f"\n⏱️ Перевірка кожні {min_interval}s ({min_interval/3600:.1f}h)")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 Ітерація #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                try:
                    # Отримуємо сигнали від всіх стратегій
                    signals = await self.strategy_manager.get_all_signals()
                    
                    if signals:
                        logger.info(f"\n📊 Знайдено {len(signals)} сигналів")
                        
                        for signal in signals:
                            await self.execute_signal(signal)
                    else:
                        logger.info("⏸️ Немає сигналів - HOLD")
                    
                    # Статистика
                    if iteration % 10 == 0:
                        self.strategy_manager.print_stats()
                    
                except Exception as e:
                    logger.error(f"❌ Помилка ітерації: {e}", exc_info=True)
                    await telegram_notifier.send_message(f"❌ ERROR: {str(e)[:100]}")
                
                # Очікування
                logger.info(f"\n⏳ Очікування {min_interval}s до наступної перевірки...")
                await asyncio.sleep(min_interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Multi-Strategy Bot зупинено")
            await telegram_notifier.send_message("👋 Multi-Strategy Bot зупинено")
        except Exception as e:
            logger.error(f"\n❌ Критична помилка: {e}", exc_info=True)
            await telegram_notifier.send_message(f"❌ CRITICAL ERROR: {str(e)[:100]}")


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description='Multi-Strategy Trading Bot')
    parser.add_argument('--enable-trading', action='store_true', help='Увімкнути реальну торгівлю')
    parser.add_argument('--all-symbols', action='store_true', help='Використовувати всі символи (включно з 60-70%)')
    parser.add_argument('--testnet', action='store_true', default=True, help='Використовувати testnet (default)')
    
    args = parser.parse_args()
    
    bot = MultiStrategyBot(
        testnet=args.testnet,
        enable_trading=args.enable_trading,
        use_all_symbols=args.all_symbols
    )
    
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
