#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 TRADING BOT - Головна точка входу
Автоматична торгівля з ML прогнозами та аналітикою
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
import logging

from core import TradingBot, get_analytics, TradingSession
from config import TRADING_CONFIG, MTF_CONFIG, WEBSOCKET_CONFIG
from telegram_bot import telegram_notifier

# Логування
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Головна функція"""
    
    # 🔍 PREFLIGHT CHECK
    logger.info("\n" + "="*80)
    logger.info("🔍 ЗАПУСК PREFLIGHT CHECK")
    logger.info("="*80 + "\n")
    
    try:
        from preflight_check import preflight_check
        
        success = await preflight_check()
        
        if not success:
            logger.error("🚫 Preflight check провалений - запуск скасовано")
            logger.info("\n💡 Виправте помилки і запустіть знову")
            await telegram_notifier.send_message(
                "🚫 Bot startup cancelled\n❌ Preflight check failed"
            )
            return
        
        logger.info("\n✅ Preflight check пройдено - продовжуємо запуск\n")
        await asyncio.sleep(2)  # Пауза для читабельності
        
    except Exception as e:
        logger.error(f"❌ Помилка preflight check: {e}")
        logger.warning("⚠️ Продовжуємо без перевірки...")
    
    # Перевірка нічного режиму
    current_hour = datetime.now().hour
    night_config = TRADING_CONFIG.get('night_mode', {})
    is_night = (
        night_config.get('enabled', False) and 
        night_config['start_hour'] <= current_hour < night_config['end_hour']
    )
    
    if is_night:
        logger.info("🌙 НІЧНИЙ РЕЖИМ - зменшена волатильність")
        interval = int(TRADING_CONFIG['check_interval'] * night_config['interval_multiplier'])
        min_conf = TRADING_CONFIG['min_confidence'] + night_config['min_confidence_boost']
    else:
        logger.info("☀️ ДЕННИЙ РЕЖИМ - нормальна волатильність")
        interval = TRADING_CONFIG['check_interval']
        min_conf = TRADING_CONFIG['min_confidence']
    
    logger.info("="*80)
    logger.info("🤖 ЗАПУСК ТОРГОВОЇ СИСТЕМИ")
    logger.info("="*80)
    logger.info(f"📅 Старт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📝 Лог файл: {log_file}")
    logger.info(f"📊 Символи: {', '.join(TRADING_CONFIG['symbols'])}")
    logger.info(f"🎯 Min confidence: {min_conf:.0%}")
    logger.info(f"💰 Position size: ${TRADING_CONFIG['position_size_usd']}")
    logger.info(f"⚡ Leverage: {TRADING_CONFIG['leverage']}x")
    logger.info(f"⏱️  Check interval: {interval}s ({interval/60:.0f} хв)")
    logger.info(f"🔄 MTF: {', '.join(MTF_CONFIG.keys())} (15m+1h)")
    logger.info(f"🔌 WebSocket: {'✅' if WEBSOCKET_CONFIG.get('enabled', False) else '❌'}")
    logger.info("="*80 + "\n")
    
    # Telegram старт
    await telegram_notifier.send_message(
        f"🤖 ТОРГОВА СИСТЕМА ЗАПУЩЕНА\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'🌙 НІЧНИЙ РЕЖИМ' if is_night else '☀️ ДЕННИЙ РЕЖИМ'}\n"
        f"\n📊 Параметри:\n"
        f"• Символів: {len(TRADING_CONFIG['symbols'])}\n"
        f"• Position: ${TRADING_CONFIG['position_size_usd']} ({TRADING_CONFIG['leverage']}x)\n"
        f"• Min confidence: {min_conf:.0%}\n"
        f"• Interval: {interval/60:.0f} хв\n"
        f"• MTF: 15m+1h ✅\n"
        f"\n🛡️ ЗАХИСТ:\n"
        f"• Cooldown SL: {TRADING_CONFIG['cooldown_after_sl']/60:.0f} хв\n"
        f"• Cooldown TP: {TRADING_CONFIG['cooldown_after_tp']/60:.0f} хв\n"
        f"• Max втрат/день: {TRADING_CONFIG['max_daily_losses_per_symbol']}\n"
    )
    
    # Ініціалізація
    session = TradingSession()
    
    try:
        # Створення бота
        bot = TradingBot()
        bot.min_confidence = min_conf
        
        # Синхронізація з Binance API при старті
        logger.info("🔄 Синхронізація з Binance API...")
        from core.trades_synchronizer import sync_trades_on_startup
        await sync_trades_on_startup(bot.client)
        
        # Завантаження моделей
        logger.info("📦 Завантаження ML моделей...")
        bot.load_models()
        
        # Запуск WebSocket
        if WEBSOCKET_CONFIG['enabled']:
            logger.info("🔌 Запуск WebSocket...")
            await bot.websocket.start()
            await asyncio.sleep(2)
        
        # Перевірка та додавання SL/TP до існуючих позицій
        logger.info("🛡️ Перевірка захисту існуючих позицій...")
        await bot.ensure_all_positions_protected()
        
        # 🔍 ЗАПУСК POSITION MONITOR (критично важливо!)
        logger.info("🔍 Запуск Position Monitor...")
        from core.position_monitor import start_monitor
        monitor = await start_monitor(
            bot.client, 
            bot.position_manager,
            on_force_close_callback=bot.add_to_blacklist  # Додаємо в blacklist після force close
        )
        logger.info("✅ Position Monitor активний (перевірка кожні 45 секунд)")
        
        logger.info("🚀 Бот готовий, запускаю головний цикл...\n")
        
        # Запуск фонової задачі нічного перенавчання
        async def nightly_retraining_task():
            """Фонова задача для інкрементального дотренування моделей"""
            from incremental_retrain import IncrementalRetrainer
            
            retrainer = IncrementalRetrainer()
            
            while True:
                try:
                    now = datetime.now()
                    
                    # Планове дотренування о 04:00 UTC
                    if now.hour == 4 and now.minute < 30:
                        logger.info("🌙 Початок планового дотренування...")
                        await telegram_notifier.send_message("🌙 Початок планового дотренування моделей...")
                        
                        results = await retrainer.run_scheduled_retrain()
                        
                        if results:
                            successful = sum(1 for v in results.values() if v)
                            message = (
                                f"✅ Планове дотренування завершено\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"📊 Успішно: {successful}/{len(results)}\n"
                                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                            )
                            
                            failed = [s for s, v in results.items() if not v]
                            if failed:
                                message += f"\n\n⚠️ Помилки: {', '.join(failed)}"
                            
                            await telegram_notifier.send_message(message)
                        else:
                            await telegram_notifier.send_message("✅ Дотренування не потрібне - всі моделі OK")
                        
                        logger.info("✅ Планове дотренування завершено")
                        
                        # Чекати до наступної ночі (23 години)
                        await asyncio.sleep(3600 * 23)
                    else:
                        # Перевіряти кожні 30 хвилин
                        await asyncio.sleep(1800)
                        
                except Exception as e:
                    logger.error(f"❌ Помилка в задачі дотренування: {e}")
                    await asyncio.sleep(3600)
        
        # Запуск фонової задачі
        retraining_task = asyncio.create_task(nightly_retraining_task())
        
        # Головний цикл
        iteration = 0
        while True:
            iteration += 1
            session.iterations = iteration
            
            # Аналітика
            await get_analytics(bot, session, iteration)
            
            # Торгова ітерація
            await bot.run_iteration()
            
            # Пауза
            await asyncio.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Торгівля зупинена (Ctrl+C)")
        
        # Фінальна аналітика
        await get_analytics(bot, session, iteration)
        
        await telegram_notifier.send_message(
            f"🛑 ТОРГІВЛЯ ЗУПИНЕНА\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"⏱️ Тривалість: {session.duration()}\n"
            f"📊 Ітерацій: {session.iterations}\n"
            f"👤 Зупинено користувачем"
        )
    
    except Exception as e:
        logger.error(f"\n❌ КРИТИЧНА ПОМИЛКА: {e}", exc_info=True)
        await telegram_notifier.send_message(
            f"❌ КРИТИЧНА ПОМИЛКА\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🔥 {str(e)[:200]}\n"
            f"⏱️ Тривалість: {session.duration()}\n"
        )
        raise
    
    finally:
        # Скасування фонової задачі
        try:
            if 'retraining_task' in locals():
                retraining_task.cancel()
                logger.info("🛑 Фонова задача перенавчання зупинена")
        except Exception as e:
            logger.error(f"⚠️ Помилка зупинки задачі: {e}")
        
        # Збереження історії навчання
        try:
            # Очищення завершено - більше не потрібне
            pass
        except Exception as e:
            logger.error(f"❌ Помилка очищення: {e}")
        
        # Зупинка WebSocket
        if WEBSOCKET_CONFIG['enabled'] and bot.websocket.is_running:
            logger.info("🔌 Зупинка WebSocket...")
            await bot.websocket.stop()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 Завершення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️ Тривалість: {session.duration()}")
        logger.info(f"📊 Ітерацій: {session.iterations}")
        logger.info(f"📝 Лог: {log_file}")
        logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ББ!")
    except Exception as e:
        import traceback
        print(f"\n❌ Фатальна помилка: {e}")
        traceback.print_exc()
        sys.exit(1)
