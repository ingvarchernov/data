#!/usr/bin/env python3
"""Тест оптимізованого менеджера БД"""
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_db_manager():
    """Тестування OptimizedDatabaseManager"""
    from optimized.database import DatabaseConnection as OptimizedDatabaseManager
    
    db = OptimizedDatabaseManager()
    
    try:
        # 1. Ініціалізація
        logger.info("=" * 60)
        logger.info("1️⃣ ТЕСТ ІНІЦІАЛІЗАЦІЇ")
        logger.info("=" * 60)
        await db.initialize()
        
        # 2. Тест symbol/interval ID
        logger.info("\n" + "=" * 60)
        logger.info("2️⃣ ТЕСТ SYMBOL/INTERVAL ID")
        logger.info("=" * 60)
        
        symbol_id = await db.get_or_create_symbol_id('BTCUSDT')
        logger.info(f"✅ Symbol ID для BTCUSDT: {symbol_id}")
        
        interval_id = await db.get_or_create_interval_id('1h')
        logger.info(f"✅ Interval ID для 1h: {interval_id}")
        
        # 3. Тест отримання даних
        logger.info("\n" + "=" * 60)
        logger.info("3️⃣ ТЕСТ ОТРИМАННЯ ДАНИХ")
        logger.info("=" * 60)
        
        df = await db.get_historical_data_optimized(symbol_id, interval_id, days_back=7)
        logger.info(f"✅ Отримано {len(df)} записів")
        if not df.empty:
            logger.info(f"   Колонки: {list(df.columns)}")
            logger.info(f"   Період: {df['timestamp'].min()} - {df['timestamp'].max()}")
        
        # 4. Тест кешу
        logger.info("\n" + "=" * 60)
        logger.info("4️⃣ ТЕСТ КЕШУВАННЯ")
        logger.info("=" * 60)
        
        cache_stats = await db.get_cache_stats()
        logger.info(f"📊 Статистика кешу:")
        for key, value in cache_stats.items():
            logger.info(f"   {key}: {value}")
        
        # 5. Тест збереження сигналу
        logger.info("\n" + "=" * 60)
        logger.info("5️⃣ ТЕСТ ТОРГОВИХ СИГНАЛІВ")
        logger.info("=" * 60)
        
        # Use db.save_signal instead of deprecated save_trading_signal
        
        signal_data = {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'confidence': 0.85,
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0,
            'quantity': 0.1,
            'strategy': 'test_strategy',
            'notes': 'Тестовий сигнал'
        }
        
        signal_id = await db.save_signal(signal_data)
        logger.info(f"✅ Створено тестовий сигнал ID: {signal_id}")
        
        # 6. Тест інвалідації кешу
        logger.info("\n" + "=" * 60)
        logger.info("6️⃣ ТЕСТ ІНВАЛІДАЦІЇ КЕШУ")
        logger.info("=" * 60)
        
        await db.invalidate_cache_pattern("*")
        logger.info("✅ Кеш очищено")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ВСІ ТЕСТИ ПРОЙДЕНО")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Помилка тесту: {e}", exc_info=True)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(test_db_manager())