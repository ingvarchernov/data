# -*- coding: utf-8 -*-
"""
Оптимізований модуль для роботи з базою даних
Включає пул з'єднань, кешування, batch операції та асинхронність
"""
import asyncio
import logging
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, select, insert, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import redis
from cachetools import TTLCache
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class OptimizedDatabaseManager:
    async def get_or_create_interval_id(self, interval: str) -> int:
        """
        Отримати або створити interval_id для інтервалу у таблиці intervals
        """
        async with self.async_session_factory() as session:
            # Перевіряємо чи існує інтервал
            result = await session.execute(text("SELECT interval_id FROM intervals WHERE interval = :interval"), {"interval": interval})
            row = result.fetchone()
            if row:
                return row[0]
            # Якщо не існує — створюємо
            result = await session.execute(text("INSERT INTO intervals (interval) VALUES (:interval) RETURNING interval_id"), {"interval": interval})
            new_id = result.fetchone()[0]
            await session.commit()
            return new_id
    async def get_or_create_symbol_id(self, symbol: str) -> int:
        """
        Отримати або створити symbol_id символу у таблиці symbols
        """
        async with self.async_session_factory() as session:
            # Перевіряємо чи існує символ
            result = await session.execute(text("SELECT symbol_id FROM symbols WHERE symbol = :symbol"), {"symbol": symbol})
            row = result.fetchone()
            if row:
                return row[0]
            # Якщо не існує — створюємо
            result = await session.execute(text("INSERT INTO symbols (symbol) VALUES (:symbol) RETURNING symbol_id"), {"symbol": symbol})
            new_id = result.fetchone()[0]
            await session.commit()
            return new_id
    """Оптимізований менеджер бази даних з кешуванням та пулом з'єднань"""
    
    def __init__(self, 
                 db_url: str = None,
                 redis_url: str = "redis://localhost:6379",
                 use_redis: bool = True,
                 pool_size: int = 20,
                 max_overflow: int = 30,
                 cache_ttl: int = 3600):
        
        # Налаштування бази даних
        if db_url is None:
            env_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
            missing = [var for var in env_vars if not os.getenv(var)]
            if missing:
                raise ValueError(f"❌ Відсутні змінні середовища для підключення до БД: {missing}. Будь ласка, задайте їх у .env або через export.")
            db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        
        # Синхронний engine
        self.sync_engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        # Асинхронний engine
        async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Кешування
        self.use_redis = use_redis
        self.memory_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.redis_pool = None
        
        if use_redis:
            try:
                self.redis_client = redis.from_url(redis_url, max_connections=20, decode_responses=False)
                # Тестуємо з'єднання (через event loop)
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.redis_client.ping())
                except Exception as e:
                    logger.warning(f"⚠️ Redis ping не вдалося: {e}")
                logger.info("✅ Redis підключено успішно")
            except Exception as e:
                logger.warning(f"⚠️ Redis недоступний, використовується тільки memory cache: {e}")
                self.use_redis = False
        
        # Метадані для швидкого доступу до таблиць
        self.metadata = MetaData()
        self._load_table_metadata()
        
    def _load_table_metadata(self):
        """Завантаження метаданих таблиць"""
        try:
            self.metadata.reflect(bind=self.sync_engine)
            logger.info(f"✅ Завантажено метадані для {len(self.metadata.tables)} таблиць")
        except Exception as e:
            logger.error(f"❌ Помилка завантаження метаданих: {e}")
    
    def _generate_cache_key(self, query: str, params: Dict = None) -> str:
        """Генерація ключа для кешування"""
        key_data = f"{query}_{params or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Отримання даних з кешу"""
        # Спочатку перевіряємо memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Потім Redis
        if self.use_redis:
            try:
                data = await self.redis_client.get(cache_key)
                if data:
                    result = pickle.loads(data)
                    # Зберігаємо в memory cache для швидкого доступу
                    self.memory_cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Помилка читання з Redis: {e}")
        
        return None
    
    async def _set_cache(self, cache_key: str, data: Any, ttl: int = 3600):
        """Збереження даних в кеш"""
        # Memory cache
        self.memory_cache[cache_key] = data
        
        # Redis cache
        if self.use_redis:
            try:
                serialized_data = pickle.dumps(data)
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            except Exception as e:
                logger.warning(f"⚠️ Помилка запису в Redis: {e}")
    
    async def execute_query_cached(self, 
                                  query: str, 
                                  params: Dict = None, 
                                  use_cache: bool = True,
                                  cache_ttl: int = 3600) -> pd.DataFrame:
        """Виконання запиту з кешуванням"""
        cache_key = self._generate_cache_key(query, params) if use_cache else None
        
        # Перевіряємо кеш
        if use_cache and cache_key:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"🎯 Cache hit для запиту: {query[:50]}...")
                return cached_result
        
        # Виконуємо запит
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # Зберігаємо в кеш
                if use_cache and cache_key:
                    await self._set_cache(cache_key, df, cache_ttl)
                    logger.debug(f"💾 Результат збережено в кеш")
                
                return df
                
            except Exception as e:
                logger.error(f"❌ Помилка виконання запиту: {e}")
                await session.rollback()
                raise
    
    async def batch_insert(self, table_name: str, data: List[Dict], batch_size: int = 1000) -> bool:
        """Пакетна вставка даних"""
        if not data:
            return True
            
        table = self.metadata.tables.get(table_name)
        if table is None:
            logger.error(f"❌ Таблиця {table_name} не знайдена")
            return False
        
        async with self.async_session_factory() as session:
            try:
                # Розбиваємо на батчі
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    await session.execute(insert(table), batch)
                    
                await session.commit()
                logger.info(f"✅ Вставлено {len(data)} записів в {table_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Помилка пакетної вставки в {table_name}: {e}")
                await session.rollback()
                return False
    
    async def batch_upsert(self, table_name: str, data: List[Dict], conflict_columns: List[str], batch_size: int = 1000) -> bool:
        """Пакетний upsert (INSERT ... ON CONFLICT UPDATE)"""
        if not data:
            return True
            
        async with self.async_session_factory() as session:
            try:
                # Формуємо запит з ON CONFLICT
                columns = list(data[0].keys())
                values_placeholder = ", ".join([f":{col}" for col in columns])
                update_set = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_columns])
                conflict_cols = ", ".join(conflict_columns)
                
                query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({values_placeholder})
                ON CONFLICT ({conflict_cols})
                DO UPDATE SET {update_set}
                """
                
                # Виконуємо батчами
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    await session.execute(text(query), batch)
                    
                await session.commit()
                logger.info(f"✅ Upsert {len(data)} записів в {table_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Помилка batch upsert в {table_name}: {e}")
                await session.rollback()
                return False
    
    async def get_historical_data_optimized(self, 
                                          symbol_id: int, 
                                          interval_id: int, 
                                          days_back: int,
                                          use_cache: bool = True) -> pd.DataFrame:
        """Оптимізоване отримання історичних даних"""
        
        # Формуємо оптимізований запит з індексами
        query = """
        SELECT 
            hd.data_id,
            hd.timestamp,
            hd.open,
            hd.high,
            hd.low,
            hd.close,
            hd.volume,
            hd.quote_av,
            hd.trades,
            hd.tb_base_av,
            hd.tb_quote_av
        FROM historical_data hd
        WHERE hd.symbol_id = :symbol_id 
        AND hd.interval_id = :interval_id
        AND hd.timestamp >= :start_time
        ORDER BY hd.timestamp ASC
        """
        
        start_time = datetime.now() - timedelta(days=days_back)
        params = {
            'symbol_id': symbol_id,
            'interval_id': interval_id,
            'start_time': start_time
        }
        
        return await self.execute_query_cached(query, params, use_cache)
    
    async def get_technical_indicators_batch(self, data_ids: List[int]) -> pd.DataFrame:
        """Пакетне отримання технічних індикаторів"""
        if not data_ids:
            return pd.DataFrame()
        
        # Формуємо запит з IN clause для пакетного отримання
        placeholders = ", ".join([":id" + str(i) for i in range(len(data_ids))])
        params = {f"id{i}": data_id for i, data_id in enumerate(data_ids)}
        
        query = f"""
        SELECT * FROM technical_indicators 
        WHERE data_id IN ({placeholders})
        ORDER BY data_id
        """
        
        return await self.execute_query_cached(query, params, use_cache=False)
    
    async def invalidate_cache_pattern(self, pattern: str):
        """Інвалідація кешу за паттерном"""
        # Очищуємо memory cache (всі ключі)
        if pattern == "*":
            self.memory_cache.clear()
        
        # Очищуємо Redis за паттерном
        if self.use_redis:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"🗑️ Очищено {len(keys)} ключів з Redis")
            except Exception as e:
                logger.warning(f"⚠️ Помилка очищення Redis кешу: {e}")
    
    async def get_cache_stats(self) -> Dict:
        """Статистика кешування"""
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_hits': getattr(self.memory_cache, 'hits', 0),
            'memory_cache_misses': getattr(self.memory_cache, 'misses', 0)
        }
        
        if self.use_redis:
            try:
                redis_info = self.redis_client.info('memory')
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"⚠️ Помилка отримання статистики Redis: {e}")
        
        return stats
    
    async def close(self):
        """Закриття з'єднань"""
        await self.async_engine.dispose()
        self.sync_engine.dispose()
        
        if self.redis_pool:
            self.redis_pool.disconnect()

# Глобальний менеджер БД
db_manager = OptimizedDatabaseManager()

# Швидкі функції для зворотної сумісності
async def get_historical_data_from_db_async(symbol: str, interval: str, days_back: int, api_key: str = None, api_secret: str = None, skip_append: bool = False) -> pd.DataFrame:
    """Асинхронне отримання історичних даних"""
    # Отримуємо ID символу та інтервалу
    symbol_id = await get_or_create_symbol_id(symbol)
    interval_id = await get_or_create_interval_id(interval)
    
    return await db_manager.get_historical_data_optimized(symbol_id, interval_id, days_back)

async def get_or_create_symbol_id(symbol: str) -> int:
    """Отримання або створення ID символу"""
    query = "INSERT INTO symbols (symbol) VALUES (:symbol) ON CONFLICT (symbol) DO NOTHING RETURNING symbol_id"
    result = await db_manager.execute_query_cached(query, {'symbol': symbol}, use_cache=True, cache_ttl=86400)
    
    if result.empty:
        # Символ вже існує, отримуємо його ID
        query = "SELECT symbol_id FROM symbols WHERE symbol = :symbol"
        result = await db_manager.execute_query_cached(query, {'symbol': symbol}, use_cache=True, cache_ttl=86400)
    
    return int(result.iloc[0]['symbol_id'])

async def get_or_create_interval_id(interval: str) -> int:
    """Отримання або створення ID інтервалу"""
    query = "INSERT INTO intervals (interval_name) VALUES (:interval) ON CONFLICT (interval_name) DO NOTHING RETURNING interval_id"
    result = await db_manager.execute_query_cached(query, {'interval': interval}, use_cache=True, cache_ttl=86400)
    
    if result.empty:
        # Інтервал вже існує, отримуємо його ID
        query = "SELECT interval_id FROM intervals WHERE interval_name = :interval"
        result = await db_manager.execute_query_cached(query, {'interval': interval}, use_cache=True, cache_ttl=86400)
    
    return int(result.iloc[0]['interval_id'])

async def batch_insert_technical_indicators(indicators_data: List[Dict]) -> bool:
    """Пакетна вставка технічних індикаторів"""
    return await db_manager.batch_insert('technical_indicators', indicators_data)

async def batch_insert_historical_data(historical_data: List[Dict]) -> bool:
    """Пакетна вставка історичних даних"""
    return await db_manager.batch_upsert(
        'historical_data', 
        historical_data, 
        ['symbol_id', 'interval_id', 'timestamp']
    )

# Синхронні обгортки для зворотної сумісності
def get_historical_data_from_db(symbol: str, interval: str, days_back: int, api_key: str = None, api_secret: str = None, skip_append: bool = False) -> pd.DataFrame:
    """Синхронна обгортка для отримання історичних даних"""
    return asyncio.run(get_historical_data_from_db_async(symbol, interval, days_back, api_key, api_secret, skip_append))

def insert_symbol(symbol: str) -> int:
    """Синхронна обгортка для вставки символу"""
    return asyncio.run(get_or_create_symbol_id(symbol))

def insert_interval(interval: str) -> int:
    """Синхронна обгортка для вставки інтервалу"""
    return asyncio.run(get_or_create_interval_id(interval))