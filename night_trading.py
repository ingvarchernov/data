#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌙 НІЧНИЙ РЕЖИМ ТОРГІВЛІ
Оптимізований для довгострокового запуску з мінімальним втручанням
"""
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_trading_bot import SimpleTradingBot
from telegram_bot import telegram_notifier
import logging

# Розширене логування у файл + консоль
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'night_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфігурація для нічної торгівлі
NIGHT_CONFIG = {
    'symbols': [
        'BTCUSDT',    # 81% accuracy - ТОП
        'ETHUSDT',    # Висока ліквідність
        'BNBUSDT',    # Стабільна
        'ADAUSDT',    # Добра точність
        'XRPUSDT',    # Ліквідна
        'DOTUSDT',    # Перевірена
    ],
    'min_confidence': 0.75,      # 75% впевненість (баланс між якістю та частотою)
    'position_size_usd': 50.0,   # $50 на позицію
    'leverage': 25,              # 25x плече
    'check_interval': 900,       # 15 хвилин (900 сек) - оптимально для 4h timeframe
    'use_mtf': True,             # Multi-timeframe аналіз
    'enable_websocket': True,    # Real-time оновлення ордерів
}

async def main():
    """Головна функція нічної торгівлі"""
    
    logger.info("="*80)
    logger.info("🌙 ЗАПУСК НІЧНОГО РЕЖИМУ ТОРГІВЛІ")
    logger.info("="*80)
    logger.info(f"📅 Старт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📝 Лог файл: {log_file}")
    logger.info(f"📊 Символи: {', '.join(NIGHT_CONFIG['symbols'])}")
    logger.info(f"🎯 Min confidence: {NIGHT_CONFIG['min_confidence']:.0%}")
    logger.info(f"💰 Position size: ${NIGHT_CONFIG['position_size_usd']}")
    logger.info(f"⚡ Leverage: {NIGHT_CONFIG['leverage']}x")
    logger.info(f"⏱️  Check interval: {NIGHT_CONFIG['check_interval']}s ({NIGHT_CONFIG['check_interval']/60:.0f} хв)")
    logger.info(f"🔄 MTF Analysis: {'✅' if NIGHT_CONFIG['use_mtf'] else '❌'}")
    logger.info(f"🔌 WebSocket: {'✅' if NIGHT_CONFIG['enable_websocket'] else '❌'}")
    logger.info("="*80 + "\n")
    
    # Telegram notification про старт
    await telegram_notifier.send_message(
        f"🌙 НІЧНА ТОРГІВЛЯ РОЗПОЧАТА\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📊 Символів: {len(NIGHT_CONFIG['symbols'])}\n"
        f"💰 Position: ${NIGHT_CONFIG['position_size_usd']} ({NIGHT_CONFIG['leverage']}x)\n"
        f"🎯 Min confidence: {NIGHT_CONFIG['min_confidence']:.0%}\n"
        f"⏱️ Interval: {NIGHT_CONFIG['check_interval']/60:.0f} хв\n"
        f"🔄 MTF: {'✅' if NIGHT_CONFIG['use_mtf'] else '❌'}\n"
        f"\n🛡️ ЗАХИСТ:\n"
        f"• Cooldown SL: 1 год\n"
        f"• Cooldown TP: 30 хв\n"
        f"• Max втрат/день: 3\n"
        f"\n⚠️ БЕЗ БД - всі дані в пам'яті"
    )
    
    try:
        # Ініціалізація бота
        bot = SimpleTradingBot(
            symbols=NIGHT_CONFIG['symbols'],
            testnet=True,
            enable_trading=True  # ✅ РЕАЛЬНА ТОРГІВЛЯ на testnet
        )
        
        # Застосування конфігурації
        bot.min_confidence = NIGHT_CONFIG['min_confidence']
        bot.position_size_usd = NIGHT_CONFIG['position_size_usd']
        bot.leverage = NIGHT_CONFIG['leverage']
        bot.use_mtf = NIGHT_CONFIG['use_mtf']
        
        # Запуск бота
        logger.info("🚀 Бот ініціалізовано, запускаю головний цикл...\n")
        await bot.run(interval_seconds=NIGHT_CONFIG['check_interval'])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Нічна торгівля зупинена користувачем (Ctrl+C)")
        await telegram_notifier.send_message(
            f"🛑 НІЧНА ТОРГІВЛЯ ЗУПИНЕНА\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"👤 Зупинено користувачем"
        )
        
    except Exception as e:
        logger.error(f"\n❌ КРИТИЧНА ПОМИЛКА: {e}", exc_info=True)
        await telegram_notifier.send_message(
            f"❌ КРИТИЧНА ПОМИЛКА\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🔥 {str(e)[:200]}\n"
            f"\n⚠️ Бот зупинено"
        )
        raise
    
    finally:
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 Завершення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📝 Лог збережено: {log_file}")
        logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До побачення!")
    except Exception as e:
        print(f"\n❌ Фатальна помилка: {e}")
        sys.exit(1)
