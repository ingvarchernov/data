"""
Database Cache Management

Redis та memory caching для database queries:
- Redis cache (distributed)
- TTL cache (in-memory fallback)
- Cache key generation
- Serialization/deserialization
"""

import logging
import pickle
import hashlib
import json
from typing import Any, Optional
import redis.asyncio as redis
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Управління кешуванням
    
    Features:
    - Redis cache (distributed, persistent)
    - Memory cache (local, fast fallback)
    - Automatic serialization
    - TTL support
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        use_redis: bool = True,
        cache_ttl: int = 3600,
        memory_cache_size: int = 1000
    ):
        """
        Ініціалізація
        
        Args:
            redis_url: Redis connection URL
            use_redis: Enable Redis caching
            cache_ttl: Cache time-to-live (seconds)
            memory_cache_size: Memory cache max size
        """
        self.use_redis = use_redis
        self.cache_ttl = cache_ttl
        
        # Memory cache (завжди доступний)
        self.memory_cache = TTLCache(maxsize=memory_cache_size, ttl=cache_ttl)
        
        # Redis client
        self.redis_client = None
        if use_redis:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    max_connections=20,
                    decode_responses=False
                )
                logger.info("✅ Redis client created")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable, using memory cache only: {e}")
                self.use_redis = False
    
    async def test_redis(self) -> bool:
        """Тестування Redis з'єднання"""
        if not self.use_redis or not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            logger.info("✅ Redis connection successful")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis ping failed: {e}")
            self.use_redis = False
            return False
    
    @staticmethod
    def generate_cache_key(query: str, params: dict = None) -> str:
        """
        Генерація cache key
        
        Args:
            query: SQL query or identifier
            params: Query parameters
        
        Returns:
            Cache key (hash)
        """
        key_data = query
        if params:
            key_data += json.dumps(params, sort_keys=True)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Отримання з кешу
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        # Спробувати Redis
        if self.use_redis and self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
        
        # Fallback до memory cache
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Збереження в кеш
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (seconds), uses default if None
        """
        if ttl is None:
            ttl = self.cache_ttl
        
        # Зберегти в Redis
        if self.use_redis and self.redis_client:
            try:
                data = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, data)
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")
        
        # Зберегти в memory cache
        self.memory_cache[key] = value
    
    async def delete(self, key: str):
        """Видалення з кешу"""
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete failed: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
    
    async def clear(self):
        """Очистка всього кешу"""
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.debug(f"Redis clear failed: {e}")
        
        self.memory_cache.clear()
        logger.info("✅ Cache cleared")
    
    async def close(self):
        """Закриття Redis з'єднання"""
        if self.redis_client:
            await self.redis_client.close()


__all__ = [
    'CacheManager',
]
