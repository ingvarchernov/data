"""
Optimized Database Package

Modules:
- connection: Connection pooling  
- cache: Redis + memory caching
"""

from optimized.database.connection import (
    DatabaseConnection,
    db_manager,
    save_trading_signal,
    save_position,
    save_trade,
)
from optimized.database.cache import CacheManager

__all__ = [
    'DatabaseConnection',
    'CacheManager',
    'db_manager',
    'save_trading_signal',
    'save_position',
    'save_trade',
]

from .connection import DatabaseConnection
from .cache import CacheManager

__all__ = [
    'DatabaseConnection',
    'CacheManager',
]
