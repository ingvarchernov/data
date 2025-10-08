# -*- coding: utf-8 -*-
"""
Універсальна система кешування з Redis та memory cache
Підтримує серіалізацію, TTL, патерни ключів та метрики
"""
import asyncio
import logging
import pickle
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps

import redis
import redis.asyncio as aioredis
import pandas as pd
import numpy as np
from cachetools import TTLCache, LRUCache

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Статистика кешування"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

class SmartSerializer:
    """Розумний серіалізатор для різних типів даних"""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Серіалізація даних"""
        try:
            if isinstance(data, pd.DataFrame):
                return pickle.dumps({
                    'type': 'dataframe',
                    'data': data.to_dict('records'),
                    'index': data.index.tolist(),
                    'columns': data.columns.tolist()
                })
            elif isinstance(data, pd.Series):
                return pickle.dumps({
                    'type': 'series',
                    'data': data.to_dict(),
                    'name': data.name
                })
            elif isinstance(data, np.ndarray):
                return pickle.dumps({
                    'type': 'numpy',
                    'data': data.tolist(),
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                })
            elif isinstance(data, (dict, list, str, int, float, bool)):
                return pickle.dumps({
                    'type': 'simple',
                    'data': data
                })
            else:
                return pickle.dumps({
                    'type': 'complex',
                    'data': data
                })
        except Exception as e:
            logger.error(f"Помилка серіалізації: {e}")
            raise
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Десеріалізація даних"""
        try:
            obj = pickle.loads(data)
            
            if obj['type'] == 'dataframe':
                df = pd.DataFrame(obj['data'])
                if obj['index']:
                    df.index = obj['index']
                return df
            elif obj['type'] == 'series':
                return pd.Series(obj['data'], name=obj['name'])
            elif obj['type'] == 'numpy':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif obj['type'] in ['simple', 'complex']:
                return obj['data']
            else:
                return obj['data']
        except Exception as e:
            logger.error(f"Помилка десеріалізації: {e}")
            raise

class AdvancedCacheManager:
    """Розширений менеджер кешування"""
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 memory_cache_size: int = 1000,
                 default_ttl: int = 3600,
                 use_redis: bool = True,
                 use_memory: bool = True,
                 key_prefix: str = "crypto_ml"):
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.use_redis = use_redis
        self.use_memory = use_memory
        self.key_prefix = key_prefix
        
        # Статистика
        self.stats = CacheStats()
        
        # Memory cache
        if use_memory:
            self.memory_cache = TTLCache(maxsize=memory_cache_size, ttl=default_ttl)
            self.lru_cache = LRUCache(maxsize=memory_cache_size // 2)
        
        # Redis connection
        self.redis_client = None
        self.async_redis = None
        
        if use_redis:
            self._init_redis()
    
    def _init_redis(self):
        """Ініціалізація Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Тестуємо з'єднання
            self.redis_client.ping()
            logger.info("✅ Redis підключено успішно")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis недоступний, використовується тільки memory cache: {e}")
            self.use_redis = False
    
    async def _get_async_redis(self):
        """Отримання асинхронного Redis клієнта"""
        if not self.async_redis:
            try:
                self.async_redis = aioredis.from_url(self.redis_url)
                await self.async_redis.ping()
            except Exception as e:
                logger.warning(f"⚠️ Async Redis недоступний: {e}")
                return None
        return self.async_redis
    
    def _generate_key(self, key: str) -> str:
        """Генерація повного ключа з префіксом"""
        return f"{self.key_prefix}:{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Генерація хеша для складних ключів"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Синхронне отримання з кешу"""
        full_key = self._generate_key(key)
        
        # Спочатку memory cache
        if self.use_memory and full_key in self.memory_cache:
            self.stats.hits += 1
            return self.memory_cache[full_key]
        
        # Потім LRU cache
        if self.use_memory and full_key in self.lru_cache:
            value = self.lru_cache[full_key]
            # Переміщуємо в TTL cache
            self.memory_cache[full_key] = value
            self.stats.hits += 1
            return value
        
        # Потім Redis
        if self.use_redis and self.redis_client:
            try:
                data = self.redis_client.get(full_key)
                if data:
                    value = SmartSerializer.deserialize(data)
                    # Зберігаємо в memory cache
                    if self.use_memory:
                        self.memory_cache[full_key] = value
                    self.stats.hits += 1
                    return value
            except Exception as e:
                logger.error(f"Помилка читання з Redis: {e}")
                self.stats.errors += 1
        
        self.stats.misses += 1
        return default
    
    async def get_async(self, key: str, default: Any = None) -> Any:
        """Асинхронне отримання з кешу"""
        full_key = self._generate_key(key)
        
        # Memory cache
        if self.use_memory and full_key in self.memory_cache:
            self.stats.hits += 1
            return self.memory_cache[full_key]
        
        # Redis
        if self.use_redis:
            try:
                redis_client = await self._get_async_redis()
                if redis_client:
                    data = await redis_client.get(full_key)
                    if data:
                        value = SmartSerializer.deserialize(data)
                        # Зберігаємо в memory cache
                        if self.use_memory:
                            self.memory_cache[full_key] = value
                        self.stats.hits += 1
                        return value
            except Exception as e:
                logger.error(f"Помилка асинхронного читання з Redis: {e}")
                self.stats.errors += 1
        
        self.stats.misses += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Синхронне збереження в кеш"""
        full_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        try:
            # Memory cache
            if self.use_memory:
                self.memory_cache[full_key] = value
                self.lru_cache[full_key] = value
            
            # Redis
            if self.use_redis and self.redis_client:
                serialized_data = SmartSerializer.serialize(value)
                self.redis_client.setex(full_key, ttl, serialized_data)
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Помилка збереження в кеш: {e}")
            self.stats.errors += 1
            return False
    
    async def set_async(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Асинхронне збереження в кеш"""
        full_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        try:
            # Memory cache
            if self.use_memory:
                self.memory_cache[full_key] = value
            
            # Redis
            if self.use_redis:
                redis_client = await self._get_async_redis()
                if redis_client:
                    serialized_data = SmartSerializer.serialize(value)
                    await redis_client.setex(full_key, ttl, serialized_data)
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Помилка асинхронного збереження в кеш: {e}")
            self.stats.errors += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Видалення з кешу"""
        full_key = self._generate_key(key)
        
        try:
            # Memory cache
            if self.use_memory:
                self.memory_cache.pop(full_key, None)
                self.lru_cache.pop(full_key, None)
            
            # Redis
            if self.use_redis and self.redis_client:
                self.redis_client.delete(full_key)
            
            self.stats.deletes += 1
            return True
            
        except Exception as e:
            logger.error(f"Помилка видалення з кешу: {e}")
            self.stats.errors += 1
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Видалення за патерном"""
        deleted_count = 0
        full_pattern = self._generate_key(pattern)
        
        try:
            # Memory cache (видаляємо всі ключі що починаються з патерну)
            if self.use_memory:
                keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(full_pattern.replace('*', ''))]
                for key in keys_to_delete:
                    self.memory_cache.pop(key, None)
                    self.lru_cache.pop(key, None)
                deleted_count += len(keys_to_delete)
            
            # Redis
            if self.use_redis and self.redis_client:
                keys = self.redis_client.keys(full_pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    deleted_count += len(keys)
            
            logger.info(f"🗑️ Видалено {deleted_count} ключів за патерном {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Помилка видалення за патерном: {e}")
            return 0
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Інвалідація кешу за тегами"""
        deleted_count = 0
        for tag in tags:
            deleted_count += self.delete_pattern(f"*{tag}*")
        return deleted_count
    
    def cache_decorator(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Декоратор для кешування функцій"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Генерація ключа
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{self._hash_key({'args': args, 'kwargs': kwargs})}"
                
                # Спроба отримати з кешу
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Виконуємо функцію
                result = func(*args, **kwargs)
                
                # Зберігаємо в кеш
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def async_cache_decorator(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Асинхронний декоратор для кешування"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Генерація ключа
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{self._hash_key({'args': args, 'kwargs': kwargs})}"
                
                # Спроба отримати з кешу
                result = await self.get_async(cache_key)
                if result is not None:
                    return result
                
                # Виконуємо функцію
                result = await func(*args, **kwargs)
                
                # Зберігаємо в кеш
                await self.set_async(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict:
        """Отримання статистики кешування"""
        stats = self.stats.to_dict()
        
        if self.use_memory:
            stats['memory_cache_size'] = len(self.memory_cache)
            stats['lru_cache_size'] = len(self.lru_cache)
        
        if self.use_redis and self.redis_client:
            try:
                redis_info = self.redis_client.info('memory')
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"Помилка отримання статистики Redis: {e}")
        
        return stats
    
    def clear_all(self):
        """Очищення всього кешу"""
        if self.use_memory:
            self.memory_cache.clear()
            self.lru_cache.clear()
        
        if self.use_redis and self.redis_client:
            pattern = self._generate_key("*")
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        
        logger.info("🗑️ Весь кеш очищено")
    
    def warmup_cache(self, data_dict: Dict[str, Any], ttl: Optional[int] = None):
        """Попереднє завантаження кешу"""
        logger.info(f"🔥 Попереднє завантаження {len(data_dict)} елементів в кеш")
        
        for key, value in data_dict.items():
            self.set(key, value, ttl)
        
        logger.info("✅ Кеш завантажено")

# Глобальний менеджер кешу
cache_manager = AdvancedCacheManager()

# Зручні функції
def cache_get(key: str, default: Any = None) -> Any:
    """Швидке отримання з кешу"""
    return cache_manager.get(key, default)

def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Швидке збереження в кеш"""
    return cache_manager.set(key, value, ttl)

def cache_delete(key: str) -> bool:
    """Швидке видалення з кешу"""
    return cache_manager.delete(key)

def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Декоратор кешування"""
    return cache_manager.cache_decorator(ttl, key_func)

def async_cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Асинхронний декоратор кешування"""
    return cache_manager.async_cache_decorator(ttl, key_func)

# Спеціалізовані функції для ML
def cache_model_prediction(symbol: str, interval: str, model_hash: str, prediction: Any, ttl: int = 1800):
    """Кешування прогнозу моделі"""
    key = f"model_pred:{symbol}:{interval}:{model_hash}"
    return cache_set(key, prediction, ttl)

def get_cached_model_prediction(symbol: str, interval: str, model_hash: str) -> Any:
    """Отримання кешованого прогнозу"""
    key = f"model_pred:{symbol}:{interval}:{model_hash}"
    return cache_get(key)

def cache_indicators(symbol: str, interval: str, indicators: Dict, ttl: int = 3600):
    """Кешування технічних індикаторів"""
    key = f"indicators:{symbol}:{interval}"
    return cache_set(key, indicators, ttl)

def get_cached_indicators(symbol: str, interval: str) -> Optional[Dict]:
    """Отримання кешованих індикаторів"""
    key = f"indicators:{symbol}:{interval}"
    return cache_get(key)

def invalidate_symbol_cache(symbol: str):
    """Інвалідація всього кешу для символу"""
    cache_manager.delete_pattern(f"*{symbol}*")

def get_cache_info() -> Dict:
    """Інформація про кеш"""
    return cache_manager.get_stats()