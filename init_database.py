#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗄️ DATABASE INITIALIZATION
Ініціалізація бази даних для торгової системи
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_db_config():
    """Отримання конфігурації БД з environment"""
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        # Парсинг DATABASE_URL (формат: postgresql://user:password@host:port/dbname)
        from urllib.parse import urlparse, unquote
        
        # Якщо пароль містить спецсимволи, вони мають бути закодовані
        # Але якщо вони не закодовані, обробляємо вручну
        try:
            parsed = urlparse(db_url)
            
            # Якщо парсинг порту зламався через спецсимволи в паролі
            # Парсимо вручну
            if '@' in db_url:
                # postgresql://user:password@host:port/dbname
                protocol_part, rest = db_url.split('://', 1)
                
                # Розділяємо на credentials та host_db
                if '@' in rest:
                    credentials, host_db = rest.rsplit('@', 1)
                    
                    # Розділяємо user:password
                    if ':' in credentials:
                        user, password = credentials.split(':', 1)
                    else:
                        user = credentials
                        password = ''
                    
                    # Розділяємо host:port/dbname
                    if '/' in host_db:
                        host_port, database = host_db.split('/', 1)
                    else:
                        host_port = host_db
                        database = 'trading_db'
                    
                    # Розділяємо host:port
                    if ':' in host_port:
                        host, port = host_port.split(':', 1)
                        port = int(port)
                    else:
                        host = host_port
                        port = 5432
                    
                    return {
                        'host': host,
                        'port': port,
                        'database': database,
                        'user': user,
                        'password': password
                    }
            
            # Якщо все впорядку - використовуємо стандартний парсинг
            return {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path[1:],
                'user': parsed.username,
                'password': parsed.password
            }
        except Exception as e:
            logger.error(f"❌ Помилка парсингу DATABASE_URL: {e}")
            logger.info("💡 Використовуйте окремі змінні або закодуйте спецсимволи")
            raise
    else:
        # Окремі змінні
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'trading_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }


def check_database_exists(config):
    """Перевірка чи існує база даних"""
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database='postgres',  # З'єднуємося з дефолтною БД
            user=config['user'],
            password=config['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (config['database'],)
        )
        exists = cursor.fetchone() is not None
        
        cursor.close()
        conn.close()
        
        return exists
    except Exception as e:
        logger.error(f"❌ Помилка перевірки БД: {e}")
        return False


def create_database(config):
    """Створення бази даних"""
    try:
        logger.info(f"📝 Створення бази даних '{config['database']}'...")
        
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database='postgres',
            user=config['user'],
            password=config['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(f"CREATE DATABASE {config['database']}")
        
        cursor.close()
        conn.close()
        
        logger.info(f"✅ База даних '{config['database']}' створена")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка створення БД: {e}")
        return False


def drop_all_tables(config):
    """Видалення всіх таблиць з БД (без видалення самої БД)"""
    try:
        logger.info(f"🗑️  Видалення всіх таблиць з БД '{config['database']}'...")
        
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        # Отримуємо список всіх таблиць
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        if tables:
            logger.info(f"   Знайдено таблиць: {len(tables)}")
            
            # Видаляємо всі таблиці з CASCADE
            for table in tables:
                logger.info(f"   🗑️  Видалення таблиці: {table}")
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            
            conn.commit()
            logger.info("✅ Всі таблиці видалено")
        else:
            logger.info("   ℹ️  Таблиць не знайдено")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"❌ Помилка видалення таблиць: {e}")
        return False


def execute_schema(config, schema_file='database_schema.sql'):
    """Виконання SQL схеми"""
    try:
        schema_path = Path(__file__).parent / schema_file
        
        if not schema_path.exists():
            logger.error(f"❌ Файл схеми не знайдено: {schema_path}")
            return False
        
        logger.info(f"📝 Виконання схеми з {schema_file}...")
        
        # Читаємо SQL
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Виконуємо
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        cursor.execute(sql_script)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Схема виконана успішно")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка виконання схеми: {e}")
        return False


def verify_tables(config):
    """Перевірка створених таблиць"""
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        cursor = conn.cursor()
        
        # Отримуємо список таблиць
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        logger.info("\n📊 Створені таблиці:")
        for table in tables:
            # Отримуємо кількість записів
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"   ✅ {table:25} ({count} записів)")
        
        cursor.close()
        conn.close()
        
        return len(tables) > 0
    except Exception as e:
        logger.error(f"❌ Помилка перевірки таблиць: {e}")
        return False


def initialize_database(force_recreate=False, drop_tables_only=False):
    """Повна ініціалізація бази даних"""
    logger.info("="*70)
    logger.info("🗄️  ІНІЦІАЛІЗАЦІЯ БАЗИ ДАНИХ")
    logger.info("="*70)
    
    # Отримуємо конфігурацію
    config = get_db_config()
    
    logger.info(f"\n📝 Конфігурація:")
    logger.info(f"   Host: {config['host']}:{config['port']}")
    logger.info(f"   Database: {config['database']}")
    logger.info(f"   User: {config['user']}")
    
    # Перевіряємо чи існує БД
    db_exists = check_database_exists(config)
    
    if db_exists:
        if drop_tables_only:
            # Видаляємо тільки таблиці (БД залишається)
            logger.warning(f"\n⚠️  Видалення всіх таблиць з БД '{config['database']}'!")
            response = input("Продовжити? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("❌ Операцію скасовано")
                return False
            
            if not drop_all_tables(config):
                return False
        elif force_recreate:
            # Видаляємо всю БД
            logger.warning(f"\n⚠️  База даних '{config['database']}' вже існує!")
            response = input("Видалити та створити заново? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("❌ Ініціалізацію скасовано")
                return False
            
            # Видалення БД
            try:
                conn = psycopg2.connect(
                    host=config['host'],
                    port=config['port'],
                    database='postgres',
                    user=config['user'],
                    password=config['password']
                )
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()
                
                # Закриваємо всі з'єднання
                cursor.execute(f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{config['database']}'
                    AND pid <> pg_backend_pid()
                """)
                
                cursor.execute(f"DROP DATABASE {config['database']}")
                cursor.close()
                conn.close()
                
                logger.info(f"🗑️  База даних '{config['database']}' видалена")
                db_exists = False
            except Exception as e:
                logger.error(f"❌ Помилка видалення БД: {e}")
                return False
        else:
            logger.info(f"\n✅ База даних '{config['database']}' вже існує")
            logger.info("   Використовуйте --force для повного перестворення")
            logger.info("   Або --drop-tables для видалення тільки таблиць")
    
    # Створення БД
    if not db_exists:
        if not create_database(config):
            return False
    
    # Виконання схеми
    if not execute_schema(config):
        return False
    
    # Перевірка таблиць
    if not verify_tables(config):
        return False
    
    logger.info("\n" + "="*70)
    logger.info("✅ БАЗА ДАНИХ ГОТОВА ДО РОБОТИ!")
    logger.info("="*70)
    
    return True


def main():
    """Головна функція"""
    force_recreate = '--force' in sys.argv or '-f' in sys.argv
    drop_tables_only = '--drop-tables' in sys.argv or '--drop' in sys.argv
    
    success = initialize_database(
        force_recreate=force_recreate,
        drop_tables_only=drop_tables_only
    )
    
    if not success:
        logger.error("\n❌ Ініціалізація не вдалася")
        sys.exit(1)
    
    logger.info("\n💡 Наступні кроки:")
    logger.info("   1. Перевірте .env файл (DATABASE_URL)")
    logger.info("   2. Запустіть торгову систему: python main.py")
    logger.info("   3. Перегляньте дані: python check_db.py")
    logger.info("\n📖 Опції запуску:")
    logger.info("   python init_database.py             # Створити БД якщо не існує")
    logger.info("   python init_database.py --drop      # Видалити тільки таблиці")
    logger.info("   python init_database.py --force     # Видалити БД повністю")


if __name__ == '__main__':
    main()
