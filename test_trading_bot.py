#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовий запуск trading bot - одна ітерація
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_trading_bot import SimpleTradingBot

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_iteration():
    """Тест однієї ітерації"""
    logger.info("=" * 70)
    logger.info("🧪 ТЕСТОВИЙ ЗАПУСК TRADING BOT (1 ітерація)")
    logger.info("=" * 70)
    
    try:
        # Створення бота (мультисимволи)
        symbols = ['BTCUSDT']  # Тільки BTC для тесту
        bot = SimpleTradingBot(symbols=symbols, testnet=True, enable_trading=False)
        
        # Завантаження моделей
        logger.info("\n📦 Завантаження моделей...")
        bot.load_models()
        
        # Баланс
        logger.info("\n💰 Перевірка балансу...")
        balance = await bot.get_balance()
        
        if balance == 0:
            logger.warning("⚠️ Баланс 0 - можливо помилка API або testnet не активований")
        
        # Тестуємо кожен символ
        for symbol in symbols:
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 ТЕСТУВАННЯ: {symbol}")
            logger.info(f"{'='*70}")
            
            # Завантаження даних
            logger.info(f"\n📊 Завантаження ринкових даних для {symbol}...")
            df = await bot.get_market_data(symbol, interval='4h', limit=500)
            
            if df.empty:
                logger.error(f"❌ Не вдалось завантажити дані для {symbol}")
                continue
            
            logger.info(f"✅ Завантажено {len(df)} свічок")
            logger.info(f"   Період: {df.index[0]} - {df.index[-1]}")
            logger.info(f"   Поточна ціна: ${df['close'].iloc[-1]:.2f}")
            
            # Прогноз
            logger.info(f"\n🤖 Генерація прогнозу для {symbol}...")
            prediction = await bot.predict(symbol, df)
            
            if not prediction:
                logger.error(f"❌ Помилка прогнозу для {symbol}")
                continue
            
            logger.info("\n" + "=" * 70)
            logger.info(f"📈 РЕЗУЛЬТАТ ПРОГНОЗУ: {symbol}")
            logger.info("=" * 70)
            logger.info(f"Напрямок:      {prediction['prediction']}")
            logger.info(f"Впевненість:   {prediction['confidence']:.2%}")
            logger.info(f"Proba DOWN:    {prediction['proba_down']:.2%}")
            logger.info(f"Proba UP:      {prediction['proba_up']:.2%}")
            logger.info(f"Поточна ціна:  ${prediction['current_price']:.2f}")
            logger.info(f"Час:           {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)
            
            # Перевірка позицій
            logger.info(f"\n📊 Перевірка поточних позицій для {symbol}...")
            position = await bot.check_position(symbol)
            
            if position:
                logger.info(f"✅ Знайдено позицію:")
                logger.info(f"   Напрямок:    {position['side']}")
                logger.info(f"   Кількість:   {abs(position['amount']):.6f}")
                logger.info(f"   Вхід:        ${position['entry_price']:.2f}")
                logger.info(f"   Unrealized:  ${position['unrealized_pnl']:.2f}")
            else:
                logger.info("ℹ️ Позицій немає")
            
            # Торгова логіка
            logger.info(f"\n🎯 Аналіз торгової можливості для {symbol}...")
            
            if prediction['confidence'] >= 0.70:
                if prediction['prediction'] == 'UP' and not position:
                    logger.info("📈 СИГНАЛ: ВІДКРИТИ LONG")
                    logger.info(f"   Рекомендований вхід: ${prediction['current_price']:.2f}")
                    logger.info(f"   Впевненість: {prediction['confidence']:.2%}")
                    logger.info("   ⚠️ DEMO MODE - угода не виконується")
                elif prediction['prediction'] == 'DOWN' and position and position['side'] == 'LONG':
                    logger.info("📉 СИГНАЛ: ЗАКРИТИ LONG")
                    logger.info(f"   Поточна ціна: ${prediction['current_price']:.2f}")
                    logger.info(f"   PnL: ${position['unrealized_pnl']:.2f}")
                    logger.info("   ⚠️ DEMO MODE - угода не виконується")
                else:
                    logger.info("⏸️ УТРИМАННЯ позиції")
            else:
                logger.info(f"⏸️ ОЧІКУВАННЯ (низька впевненість: {prediction['confidence']:.2%})")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ТЕСТ ЗАВЕРШЕНО УСПІШНО")
        logger.info("=" * 70)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Переривання користувачем")
        return False
    except Exception as e:
        logger.error(f"\n❌ Помилка: {e}", exc_info=True)
        return False


async def main():
    success = await test_single_iteration()
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
