"""
Optimized Database Package

Modules:
- connection: Connection pooling + async operations
- cache: Memory caching (Redis optional for multi-bot)
- positions: CRUD operations for positions tracking
"""

from .connection import DatabaseConnection, db_manager
from .cache import CacheManager
from .positions import PositionDB

__all__ = [
    'DatabaseConnection',
    'CacheManager',
    'PositionDB',
    'db_manager',
]
