"""
Database Migration Script
Адаптація БД під нову архітектуру

Зміни:
1. Додати created_at, updated_at до positions
2. Перевірити та оновити типи колонок
3. Очистити старі тестові дані
4. Створити тригери для updated_at
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
db_url = os.getenv('DATABASE_URL')

if not db_url:
    logger.error('❌ DATABASE_URL не знайдено')
    exit(1)

engine = create_engine(db_url)


def run_migration():
    """Виконання міграції БД"""
    
    with engine.connect() as conn:
        # Старт транзакції
        trans = conn.begin()
        
        try:
            logger.info("🚀 Початок міграції БД...")
            
            # ============================================================
            # 1. ДОДАТИ created_at, updated_at до POSITIONS
            # ============================================================
            logger.info("📝 Додавання created_at та updated_at до positions...")
            
            # Перевіряємо чи існує created_at
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='positions' AND column_name='created_at'
            """))
            
            if not result.fetchone():
                # Додаємо created_at
                conn.execute(text("""
                    ALTER TABLE positions 
                    ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                """))
                
                # Заповнюємо created_at з entry_time для існуючих записів
                conn.execute(text("""
                    UPDATE positions 
                    SET created_at = COALESCE(entry_time, CURRENT_TIMESTAMP)
                    WHERE created_at IS NULL
                """))
                logger.info("✅ created_at додано")
            else:
                logger.info("ℹ️  created_at вже існує")
            
            # Перевіряємо чи існує updated_at
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='positions' AND column_name='updated_at'
            """))
            
            if not result.fetchone():
                # Додаємо updated_at
                conn.execute(text("""
                    ALTER TABLE positions 
                    ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                """))
                
                # Заповнюємо updated_at
                conn.execute(text("""
                    UPDATE positions 
                    SET updated_at = COALESCE(created_at, entry_time, CURRENT_TIMESTAMP)
                    WHERE updated_at IS NULL
                """))
                logger.info("✅ updated_at додано")
            else:
                logger.info("ℹ️  updated_at вже існує")
            
            # ============================================================
            # 2. СТВОРИТИ ТРИГЕР для updated_at (якщо не існує)
            # ============================================================
            logger.info("📝 Створення тригера для positions.updated_at...")
            
            # Перевіряємо чи існує тригер
            result = conn.execute(text("""
                SELECT tgname 
                FROM pg_trigger 
                WHERE tgname='update_positions_updated_at'
            """))
            
            if not result.fetchone():
                # Створюємо тригер
                conn.execute(text("""
                    CREATE TRIGGER update_positions_updated_at
                        BEFORE UPDATE ON positions
                        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
                """))
                logger.info("✅ Тригер створено")
            else:
                logger.info("ℹ️  Тригер вже існує")
            
            # ============================================================
            # 3. ОЧИСТИТИ ТЕСТОВІ ДАНІ (в правильному порядку через FK)
            # ============================================================
            logger.info("🗑️  Видалення тестових даних...")
            
            # Спочатку trades (залежить від positions)
            result = conn.execute(text("""
                DELETE FROM trades WHERE symbol LIKE 'TEST%'
            """))
            deleted_trades = result.rowcount
            
            # Потім positions (залежить від trading_signals)
            result = conn.execute(text("""
                DELETE FROM positions WHERE symbol LIKE 'TEST%'
            """))
            deleted_positions = result.rowcount
            
            # В кінці trading_signals
            result = conn.execute(text("""
                DELETE FROM trading_signals WHERE symbol LIKE 'TEST%'
            """))
            deleted_signals = result.rowcount
            
            logger.info(f"✅ Видалено: {deleted_trades} угод, {deleted_positions} позицій, {deleted_signals} сигналів")
            
            # ============================================================
            # 4. ОЧИСТИТИ ЗАКРИТІ ПОЗИЦІЇ (опціонально)
            # ============================================================
            logger.info("🗑️  Видалення старих закритих позицій...")
            
            result = conn.execute(text("""
                DELETE FROM positions 
                WHERE status IN ('closed', 'cancelled') 
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '7 days'
            """))
            deleted_old = result.rowcount
            logger.info(f"✅ Видалено {deleted_old} старих закритих позицій")
            
            # ============================================================
            # 5. ПЕРЕВІРКА ЦІЛІСНОСТІ
            # ============================================================
            logger.info("✅ Перевірка цілісності даних...")
            
            # Кількість записів після міграції
            result = conn.execute(text("SELECT COUNT(*) FROM positions"))
            pos_count = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM trading_signals"))
            sig_count = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM trades"))
            trades_count = result.scalar()
            
            logger.info(f"📊 Після міграції:")
            logger.info(f"   Positions: {pos_count}")
            logger.info(f"   Signals: {sig_count}")
            logger.info(f"   Trades: {trades_count}")
            
            # Коміт транзакції
            trans.commit()
            logger.info("✅ Міграція завершена успішно!")
            
        except Exception as e:
            trans.rollback()
            logger.error(f"❌ Помилка міграції: {e}")
            raise


if __name__ == '__main__':
    print("=" * 60)
    print("DATABASE MIGRATION")
    print("=" * 60)
    
    response = input("\n⚠️  Ця операція змінить структуру БД. Продовжити? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        run_migration()
        print("\n✅ Міграція завершена!")
    else:
        print("❌ Міграцію скасовано")
