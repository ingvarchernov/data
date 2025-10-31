"""
Database Cache Management

In-memory caching для database queries та ML predictions:
- TTL cache (швидкий, простий)
- Cache key generation
- Serialization підтримка

📝 NOTE: Redis можна додати пізніше для multi-bot setup:
    - Розподілений кеш між кількома інстансами
    - Shared state для координації ботів
    - Rate limiting для API calls
    Просто розкоментуйте Redis код нижче і додайте redis>=4.5.0 в requirements
"""

import logging
import hashlib
import json
from typing import Any, Optional
from cachetools import TTLCache

# Для майбутнього multi-bot setup:
# import redis.asyncio as redis
# import pickle

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Управління кешуванням (спрощена версія)
    
    Features:
    - Memory cache (швидкий, локальний)
    - Automatic TTL
    - Simple key generation
    
    💡 Для multi-bot setup додайте Redis
    """
    
    def __init__(
        self,
        cache_ttl: int = 3600,
        memory_cache_size: int = 1000
    ):
        """
        Ініціалізація
        
        Args:
            cache_ttl: Cache time-to-live (seconds)
            memory_cache_size: Memory cache max size
        """
        self.cache_ttl = cache_ttl
        
        # Memory cache (єдиний варіант зараз)
        self.memory_cache = TTLCache(maxsize=memory_cache_size, ttl=cache_ttl)
        logger.info(f"✅ Memory cache initialized (size: {memory_cache_size}, ttl: {cache_ttl}s)")
        
        # Redis відключений (можна додати пізніше)
        # self.redis_client = None
        # self.use_redis = False
    
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
        # Memory cache only (просто і швидко)
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Збереження в кеш
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (ignored for now, uses default TTL)
        """
        # Memory cache only
        self.memory_cache[key] = value
        
        # Для майбутнього Redis:
        # if ttl is None:
        #     ttl = self.cache_ttl
        # data = pickle.dumps(value)
        # await self.redis_client.setex(key, ttl, data)
    
    async def delete(self, key: str):
        """Видалення з кешу"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Для майбутнього Redis:
        # await self.redis_client.delete(key)
    
    async def clear(self):
        """Очистка всього кешу"""
        self.memory_cache.clear()
        logger.info("✅ Cache cleared")
        
        # Для майбутнього Redis:
        # await self.redis_client.flushdb()
    
    async def close(self):
        """Закриття з'єднань (для майбутнього Redis)"""
        # Для майбутнього Redis:
        # if self.redis_client:
        #     await self.redis_client.close()
        pass


__all__ = [
    'CacheManager',
]
