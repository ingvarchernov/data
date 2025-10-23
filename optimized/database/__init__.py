"""
Optimized Database Package

Модульна структура для роботи з базою даних:
- connection: Database connection management (sync + async)
- cache: Redis + memory caching
- queries: Common database queries (будуть додані)

Backward compatibility:
- Основні компоненти для зворотної сумісності з optimized_db.py
"""

from .connection import DatabaseConnection
from .cache import CacheManager

__all__ = [
    'DatabaseConnection',
    'CacheManager',
]
