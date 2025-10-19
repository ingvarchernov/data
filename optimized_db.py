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
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, select, insert, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
import redis.asyncio as redis
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OptimizedDatabaseManager:
    """Оптимізований менеджер бази даних з кешуванням та пулом з'єднань"""
    
    def __init__(
        self, 
        db_url: str = None,
        redis_url: str = "redis://localhost:6379",
        use_redis: bool = True,
        pool_size: int = 20,
        max_overflow: int = 30,
        cache_ttl: int = 3600
    ):
        """
        Ініціалізація менеджера БД
        
        Args:
            db_url: URL підключення до PostgreSQL
            redis_url: URL підключення до Redis
            use_redis: Використовувати Redis для кешування
            pool_size: Розмір пулу з'єднань
            max_overflow: Максимальна кількість додаткових з'єднань
            cache_ttl: Час життя кешу (секунди)
        """
        # Налаштування бази даних
        if db_url is None:
            env_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
            missing = [var for var in env_vars if not os.getenv(var)]
            if missing:
                raise ValueError(
                    f"❌ Відсутні змінні середовища для підключення до БД: {missing}. "
                    f"Будь ласка, задайте їх у .env або через export."
                )
            db_url = (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            )
        
        self.db_url = db_url
        
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
        self.redis_client = None
        self.cache_ttl = cache_ttl
        
        if use_redis:
            try:
                self.redis_client = redis.from_url(
                    redis_url, 
                    max_connections=20, 
                    decode_responses=False
                )
                logger.info("✅ Redis клієнт створено")
            except Exception as e:
                logger.warning(f"⚠️ Redis недоступний, використовується тільки memory cache: {e}")
                self.use_redis = False
        
        # Метадані для швидкого доступу до таблиць
        self.metadata = MetaData()
        self._metadata_loaded = False
        
        logger.info("✅ OptimizedDatabaseManager ініціалізовано")
    
    async def initialize(self):
        """Асинхронна ініціалізація (тестування з'єднань)"""
        try:
            # Тест PostgreSQL
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            logger.info("✅ PostgreSQL з'єднання успішне")
            
            # Тест Redis
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.ping()
                    logger.info("✅ Redis з'єднання успішне")
                except Exception as e:
                    logger.warning(f"⚠️ Redis ping failed: {e}")
                    self.use_redis = False
            
            # Завантаження метаданих
            self._load_table_metadata()
            
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації БД: {e}")
            raise
    
    def _load_table_metadata(self):
        """Завантаження метаданих таблиць (синхронно при ініціалізації)"""
        if self._metadata_loaded:
            return
        
        try:
            with self.sync_engine.connect() as conn:
                self.metadata.reflect(bind=conn)
            self._metadata_loaded = True
            logger.info(f"✅ Завантажено метадані {len(self.metadata.tables)} таблиць")
        except Exception as e:
            logger.warning(f"⚠️ Помилка завантаження метаданих: {e}")
    
    def _generate_cache_key(self, query: str, params: Dict = None) -> str:
        """Генерація ключа для кешування"""
        key_data = f"{query}_{params or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Отримання даних з кешу"""
        # Memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Redis cache
        if self.use_redis and self.redis_client:
            try:
                data = await self.redis_client.get(cache_key)
                if data:
                    result = pickle.loads(data)
                    self.memory_cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Помилка читання з Redis: {e}")
        
        return None
    
    async def _set_cache(self, cache_key: str, data: Any, ttl: int = None):
        """Збереження даних в кеш"""
        if ttl is None:
            ttl = self.cache_ttl
        
        # Memory cache
        self.memory_cache[cache_key] = data
        
        # Redis cache
        if self.use_redis and self.redis_client:
            try:
                serialized_data = pickle.dumps(data)
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            except Exception as e:
                logger.warning(f"⚠️ Помилка запису в Redis: {e}")
    
    async def execute_query_cached(
        self, 
        query: str, 
        params: Dict = None, 
        use_cache: bool = True,
        cache_ttl: int = None
    ) -> pd.DataFrame:
        """Виконання запиту з кешуванням"""
        cache_key = self._generate_cache_key(query, params) if use_cache else None
        
        # Перевірка кешу
        if use_cache and cache_key:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"🎯 Cache hit: {query[:50]}...")
                return cached_result
        
        # Виконання запиту
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # Збереження в кеш
                if use_cache and cache_key:
                    await self._set_cache(cache_key, df, cache_ttl)
                    logger.debug(f"💾 Збережено в кеш")
                
                return df
                
            except Exception as e:
                logger.error(f"❌ Помилка виконання запиту: {e}")
                await session.rollback()
                raise
    
    async def get_or_create_symbol_id(self, symbol: str) -> int:
        """Отримати або створити symbol_id"""
        async with self.async_session_factory() as session:
            # Перевірка існування
            result = await session.execute(
                text("SELECT symbol_id FROM symbols WHERE symbol = :symbol"),
                {"symbol": symbol}
            )
            row = result.fetchone()
            
            if row:
                return row[0]
            
            # Створення нового
            result = await session.execute(
                text("INSERT INTO symbols (symbol) VALUES (:symbol) RETURNING symbol_id"),
                {"symbol": symbol}
            )
            new_id = result.fetchone()[0]
            await session.commit()
            
            logger.debug(f"✅ Створено symbol_id={new_id} для {symbol}")
            return new_id
    
    async def get_or_create_interval_id(self, interval: str) -> int:
        """Отримати або створити interval_id"""
        async with self.async_session_factory() as session:
            # Перевірка існування
            result = await session.execute(
                text("SELECT interval_id FROM intervals WHERE interval = :interval"),
                {"interval": interval}
            )
            row = result.fetchone()
            
            if row:
                return row[0]
            
            # Створення нового
            result = await session.execute(
                text("INSERT INTO intervals (interval) VALUES (:interval) RETURNING interval_id"),
                {"interval": interval}
            )
            new_id = result.fetchone()[0]
            await session.commit()
            
            logger.debug(f"✅ Створено interval_id={new_id} для {interval}")
            return new_id
    
    async def batch_insert(
        self, 
        table_name: str, 
        data: List[Dict], 
        batch_size: int = 1000
    ) -> bool:
        """Пакетна вставка даних"""
        if not data:
            return True
        
        if not self._metadata_loaded:
            self._load_table_metadata()
        
        table = self.metadata.tables.get(table_name)
        if table is None:
            logger.error(f"❌ Таблиця {table_name} не знайдена")
            return False
        
        async with self.async_session_factory() as session:
            try:
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
    
    async def batch_upsert(
        self, 
        table_name: str, 
        data: List[Dict], 
        conflict_columns: List[str], 
        batch_size: int = 1000
    ) -> bool:
        """Пакетний upsert (INSERT ... ON CONFLICT UPDATE)"""
        if not data:
            return True
        
        async with self.async_session_factory() as session:
            try:
                columns = list(data[0].keys())
                values_placeholder = ", ".join([f":{col}" for col in columns])
                update_set = ", ".join([
                    f"{col} = EXCLUDED.{col}" 
                    for col in columns 
                    if col not in conflict_columns
                ])
                conflict_cols = ", ".join(conflict_columns)
                
                query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({values_placeholder})
                ON CONFLICT ({conflict_cols})
                DO UPDATE SET {update_set}
                """
                
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    for record in batch:
                        await session.execute(text(query), record)
                
                await session.commit()
                logger.info(f"✅ Upsert {len(data)} записів в {table_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Помилка batch upsert в {table_name}: {e}")
                await session.rollback()
                return False
    
    async def get_historical_data_optimized(
        self, 
        symbol_id: int, 
        interval_id: int, 
        days_back: int,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Оптимізоване отримання історичних даних"""
        start_time = datetime.now() - timedelta(days=days_back)
        
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
        
        params = {
            'symbol_id': symbol_id,
            'interval_id': interval_id,
            'start_time': start_time
        }
        
        df = await self.execute_query_cached(query, params, use_cache)
        
        if not df.empty:
            logger.info(
                f"🔍 Отримано {len(df)} рядків для "
                f"symbol_id={symbol_id}, interval_id={interval_id}"
            )
        else:
            logger.warning(
                f"⚠️ Порожній результат для "
                f"symbol_id={symbol_id}, interval_id={interval_id}"
            )
        
        return df
    
    async def invalidate_cache_pattern(self, pattern: str):
        """Інвалідація кешу за паттерном"""
        if pattern == "*":
            self.memory_cache.clear()
            logger.info("🗑️ Memory cache очищено")
        
        if self.use_redis and self.redis_client:
            try:
                cursor = 0
                keys_deleted = 0
                
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=100
                    )
                    
                    if keys:
                        await self.redis_client.delete(*keys)
                        keys_deleted += len(keys)
                    
                    if cursor == 0:
                        break
                
                if keys_deleted > 0:
                    logger.info(f"🗑️ Видалено {keys_deleted} ключів з Redis")
                    
            except Exception as e:
                logger.warning(f"⚠️ Помилка очищення Redis: {e}")
    
    async def get_cache_stats(self) -> Dict:
        """Статистика кешування"""
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_maxsize': self.memory_cache.maxsize,
            'redis_available': self.use_redis
        }
        
        if self.use_redis and self.redis_client:
            try:
                info = await self.redis_client.info('memory')
                stats['redis_memory_used'] = info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = await self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"⚠️ Помилка отримання статистики Redis: {e}")
        
        return stats
    
    async def close(self):
        """Закриття з'єднань"""
        try:
            await self.async_engine.dispose()
            self.sync_engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ БД з'єднання закрито")
        except Exception as e:
            logger.error(f"❌ Помилка закриття БД: {e}")


# ============================================================================
# ТОРГОВІ ФУНКЦІЇ
# ============================================================================

async def save_trading_signal(db_manager, signal_data: Dict[str, Any]) -> int:
    """Збереження торгового сигналу"""
    try:
        signal_record = {
            'symbol': signal_data['symbol'],
            'action': signal_data['action'],
            'confidence': float(signal_data['confidence']),
            'entry_price': float(signal_data['entry_price']),
            'stop_loss': float(signal_data['stop_loss']) if signal_data.get('stop_loss') else None,
            'take_profit': float(signal_data['take_profit']) if signal_data.get('take_profit') else None,
            'quantity': float(signal_data['quantity']),
            'strategy': signal_data.get('strategy', 'unknown'),
            'prediction_source': signal_data.get('prediction_source', 'technical'),
            'status': signal_data.get('status', 'generated'),
            'notes': signal_data.get('notes', '')
        }
        
        async with db_manager.async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO trading_signals 
                    (symbol, action, confidence, entry_price, stop_loss, take_profit, 
                     quantity, strategy, prediction_source, status, notes) 
                    VALUES 
                    (:symbol, :action, :confidence, :entry_price, :stop_loss, :take_profit, 
                     :quantity, :strategy, :prediction_source, :status, :notes) 
                    RETURNING id
                """),
                signal_record
            )
            signal_id = result.fetchone()[0]
            await session.commit()
            
        logger.info(f"✅ Збережено сигнал: {signal_data['symbol']} {signal_data['action']}")
        return signal_id
        
    except Exception as e:
        logger.error(f"❌ Помилка збереження сигналу: {e}")
        return None


async def save_position(db_manager, position_data: Dict[str, Any]) -> int:
    """Збереження відкритої позиції"""
    try:
        logger.info(f"🔄 Спроба збереження позиції: {position_data.get('symbol')} {position_data.get('side')}")
        
        position_record = {
            'symbol': position_data['symbol'],
            'side': position_data['side'],
            'entry_price': float(position_data['entry_price']),
            'quantity': float(position_data['quantity']),
            'stop_loss': float(position_data['stop_loss']) if position_data.get('stop_loss') else None,
            'take_profit': float(position_data['take_profit']) if position_data.get('take_profit') else None,
            'strategy': position_data.get('strategy', 'unknown'),
            'status': position_data.get('status', 'open'),
            'signal_id': position_data.get('signal_id'),
            'metadata': json.dumps(position_data.get('metadata', {}))
        }
        
        logger.info(f"📝 Position record: entry_price={position_record['entry_price']}, quantity={position_record['quantity']}")
        
        async with db_manager.async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO positions 
                    (symbol, side, entry_price, quantity, stop_loss, take_profit, 
                     strategy, status, signal_id, metadata) 
                    VALUES 
                    (:symbol, :side, :entry_price, :quantity, :stop_loss, :take_profit, 
                     :strategy, :status, :signal_id, :metadata) 
                    RETURNING id
                """),
                position_record
            )
            position_id = result.fetchone()[0]
            await session.commit()
            
        logger.info(f"✅ Збережено позицію: {position_data['symbol']} {position_data['side']} (ID: {position_id})")
        return position_id
        
    except Exception as e:
        logger.error(f"❌ Помилка збереження позиції: {e}", exc_info=True)
        return None


async def save_trade(db_manager, trade_data: Dict[str, Any]) -> int:
    """Збереження закритої угоди"""
    try:
        entry_price = float(trade_data['entry_price'])
        exit_price = float(trade_data['exit_price'])
        quantity = float(trade_data['quantity'])
        fees = float(trade_data.get('fees', 0))
        
        # Розрахунок P&L
        if entry_price == 0:
            logger.warning(f"⚠️ Entry price = 0 для {trade_data['symbol']}")
            pnl = -fees
            pnl_percentage = 0
        elif trade_data['side'] == 'LONG':
            gross_pnl = (exit_price - entry_price) * quantity
            pnl = gross_pnl - fees
            pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * quantity
            pnl = gross_pnl - fees
            pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
        
        trade_record = {
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': trade_data['entry_time'],
            'exit_time': trade_data.get('exit_time', datetime.now()),
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'strategy': trade_data.get('strategy', 'unknown'),
            'exit_reason': trade_data.get('exit_reason', 'manual'),
            'position_id': trade_data.get('position_id'),
            'signal_id': trade_data.get('signal_id'),
            'fees': fees,
            'metadata': json.dumps(trade_data.get('metadata', {}))
        }
        
        async with db_manager.async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO trades 
                    (symbol, side, entry_price, exit_price, quantity, entry_time, exit_time, 
                     pnl, pnl_percentage, strategy, exit_reason, position_id, signal_id, fees, metadata) 
                    VALUES 
                    (:symbol, :side, :entry_price, :exit_price, :quantity, :entry_time, :exit_time, 
                     :pnl, :pnl_percentage, :strategy, :exit_reason, :position_id, :signal_id, :fees, :metadata) 
                    RETURNING id
                """),
                trade_record
            )
            trade_id = result.fetchone()[0]
            await session.commit()
            
        logger.info(
            f"✅ Збережено угоду: {trade_data['symbol']} {trade_data['side']} "
            f"P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)"
        )
        return trade_id
        
    except Exception as e:
        logger.error(f"❌ Помилка збереження угоди: {e}")
        return None


# ============================================================================
# ГЛОБАЛЬНИЙ МЕНЕДЖЕР
# ============================================================================

db_manager = OptimizedDatabaseManager()


# ============================================================================
# ФУНКЦІЇ ДЛЯ ЗВОРОТНОЇ СУМІСНОСТІ
# ============================================================================

async def get_historical_data_from_db_async(
    symbol: str, 
    interval: str, 
    days_back: int, 
    api_key: str = None, 
    api_secret: str = None, 
    skip_append: bool = False
) -> pd.DataFrame:
    """Асинхронне отримання історичних даних"""
    symbol_id = await db_manager.get_or_create_symbol_id(symbol)
    interval_id = await db_manager.get_or_create_interval_id(interval)
    
    return await db_manager.get_historical_data_optimized(
        symbol_id, interval_id, days_back
    )


def get_historical_data_from_db(
    symbol: str, 
    interval: str, 
    days_back: int, 
    api_key: str = None, 
    api_secret: str = None, 
    skip_append: bool = False
) -> pd.DataFrame:
    """Синхронна обгортка"""
    return asyncio.run(
        get_historical_data_from_db_async(
            symbol, interval, days_back, api_key, api_secret, skip_append
        )
    )