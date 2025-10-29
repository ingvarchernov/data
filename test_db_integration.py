#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки інтеграції БД
"""
import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized.database.connection import DatabaseConnection, save_position, save_trade
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_db_integration():
    """Тест збереження та читання з БД"""
    
    logger.info("🚀 Тестування інтеграції з БД...")
    
    # Ініціалізація DB
    db = DatabaseConnection()
    
    # Тест підключення
    if not await db.test_connection():
        logger.error("❌ Не вдалось підключитись до БД")
        return False
    
    logger.info("✅ Підключення до БД успішне")
    
    # ========================================================
    # ТЕСТ 1: Збереження позиції
    # ========================================================
    logger.info("\n📝 ТЕСТ 1: Збереження позиції...")
    
    test_position = {
        'symbol': 'TESTUSDT',
        'side': 'LONG',
        'entry_price': 50000.0,
        'quantity': 0.001,
        'stop_loss': 49500.0,
        'take_profit': 50500.0,
        'status': 'open',
        'strategy': 'ML_4h_TEST',
        'entry_time': datetime.now(),
        'signal_id': None,
        'metadata': {
            'confidence': 0.85,
            'leverage': 25,
            'test': True
        }
    }
    
    position_id = await save_position(db, test_position)
    
    if position_id:
        logger.info(f"✅ Позиція збережена з ID: {position_id}")
    else:
        logger.error("❌ Не вдалось зберегти позицію")
        return False
    
    # ========================================================
    # ТЕСТ 2: Читання позиції
    # ========================================================
    logger.info("\n📖 ТЕСТ 2: Читання позиції...")
    
    from sqlalchemy import text
    
    async with db.async_session_factory() as session:
        result = await session.execute(
            text("SELECT * FROM positions WHERE id = :id"),
            {'id': position_id}
        )
        row = result.fetchone()
        
        if row:
            logger.info(f"✅ Знайдено позицію:")
            logger.info(f"   Symbol: {row.symbol}")
            logger.info(f"   Side: {row.side}")
            logger.info(f"   Entry: ${row.entry_price}")
            logger.info(f"   Quantity: {row.quantity}")
            logger.info(f"   Status: {row.status}")
            logger.info(f"   Created: {row.created_at}")
        else:
            logger.error("❌ Позицію не знайдено")
            return False
    
    # ========================================================
    # ТЕСТ 3: Оновлення позиції
    # ========================================================
    logger.info("\n🔄 ТЕСТ 3: Оновлення позиції...")
    
    async with db.async_session_factory() as session:
        await session.execute(
            text("UPDATE positions SET status = :status WHERE id = :id"),
            {'status': 'closed', 'id': position_id}
        )
        await session.commit()
        logger.info("✅ Статус позиції оновлено на 'closed'")
    
    # ========================================================
    # ТЕСТ 4: Збереження trade
    # ========================================================
    logger.info("\n💰 ТЕСТ 4: Збереження trade...")
    
    test_trade = {
        'symbol': 'TESTUSDT',
        'side': 'LONG',
        'entry_price': 50000.0,
        'exit_price': 50500.0,
        'quantity': 0.001,
        'entry_time': datetime.now(),
        'exit_time': datetime.now(),
        'pnl': 0.50,
        'pnl_percentage': 1.0,
        'strategy': 'ML_4h_TEST',
        'exit_reason': 'take_profit',
        'position_id': position_id,
        'signal_id': None,
        'fees': 0.01,
        'metadata': {
            'test': True
        }
    }
    
    trade_id = await save_trade(db, test_trade)
    
    if trade_id:
        logger.info(f"✅ Trade збережено з ID: {trade_id}")
    else:
        logger.error("❌ Не вдалось зберегти trade")
        return False
    
    # ========================================================
    # ТЕСТ 5: Очищення тестових даних
    # ========================================================
    logger.info("\n🗑️  ТЕСТ 5: Очищення тестових даних...")
    
    async with db.async_session_factory() as session:
        # Видаляємо trade
        result = await session.execute(
            text("DELETE FROM trades WHERE id = :id"),
            {'id': trade_id}
        )
        
        # Видаляємо position
        result = await session.execute(
            text("DELETE FROM positions WHERE id = :id"),
            {'id': position_id}
        )
        
        await session.commit()
        logger.info("✅ Тестові дані видалено")
    
    # Закриття з'єднання
    await db.close()
    
    logger.info("\n✅ ВСІ ТЕСТИ ПРОЙДЕНО УСПІШНО!")
    return True


if __name__ == '__main__':
    success = asyncio.run(test_db_integration())
    
    if success:
        print("\n" + "="*50)
        print("✅ БД інтеграція працює коректно!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ Виявлено проблеми з БД інтеграцією")
        print("="*50)
        sys.exit(1)
