"""
Backward compatibility wrapper for optimized database modules

This file provides backward compatibility for imports from 'optimized_db'.
New code should use: from optimized.database import ...
"""
import warnings
from optimized.database import DatabaseConnection, CacheManager

warnings.warn(
    "Importing from 'optimized_db' is deprecated. "
    "Use 'from optimized.database import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Main database manager
try:
    db_manager = DatabaseConnection()
except Exception:
    db_manager = None

# Legacy class name compatibility
OptimizedDatabaseManager = DatabaseConnection

# Function aliases for backward compatibility
def save_position(*args, **kwargs):
    """Legacy function - use DatabaseConnection.save_position instead"""
    if db_manager:
        return db_manager.save_position(*args, **kwargs)
    raise RuntimeError("Database manager not initialized")

def save_trading_signal(*args, **kwargs):
    """Legacy function - use DatabaseConnection.save_signal instead"""
    if db_manager:
        return db_manager.save_signal(*args, **kwargs)
    raise RuntimeError("Database manager not initialized")

__all__ = [
    'db_manager',
    'DatabaseConnection',
    'OptimizedDatabaseManager',
    'CacheManager',
    'save_position',
    'save_trading_signal',
]
