"""
Database Connection Management

Модуль управління з'єднаннями з базою даних:
- Connection pooling (sync + async)
- PostgreSQL engine setup
- Connection testing
"""

import logging
import os
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Управління з'єднаннями з PostgreSQL
    
    Features:
    - Connection pooling
    - Sync + Async engines
    - Auto-reconnect
    - Metadata reflection
    """
    
    def __init__(
        self,
        db_url: str = None,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_recycle: int = 3600
    ):
        """
        Ініціалізація
        
        Args:
            db_url: PostgreSQL connection URL
            pool_size: Pool size
            max_overflow: Max overflow connections
            pool_recycle: Connection recycle time (seconds)
        """
        # Налаштування URL
        if db_url is None:
            db_url = self._build_url_from_env()
        
        self.db_url = db_url
        
        # Синхронний engine
        self.sync_engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=pool_recycle,
            echo=False
        )
        
        # Асинхронний engine
        async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=pool_recycle,
            echo=False
        )
        
        # Session factory
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Metadata
        self.metadata = MetaData()
        self._metadata_loaded = False
        
        logger.info("✅ DatabaseConnection initialized")
    
    def _build_url_from_env(self) -> str:
        """Побудова DB URL з environment variables"""
        env_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
        missing = [var for var in env_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(
                f"❌ Missing DB environment variables: {missing}. "
                f"Please set them in .env or via export."
            )
        
        return (
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
    
    async def test_connection(self) -> bool:
        """
        Тестування з'єднання
        
        Returns:
            bool: True якщо з'єднання успішне
        """
        try:
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            logger.info("✅ PostgreSQL connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
    
    def load_metadata(self):
        """Завантаження metadata таблиць"""
        if self._metadata_loaded:
            return
        
        try:
            with self.sync_engine.connect() as conn:
                self.metadata.reflect(bind=conn)
            self._metadata_loaded = True
            logger.info(f"✅ Loaded metadata for {len(self.metadata.tables)} tables")
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
    
    def get_table(self, table_name: str):
        """
        Отримання Table object
        
        Args:
            table_name: Назва таблиці
        
        Returns:
            Table object or None
        """
        if not self._metadata_loaded:
            self.load_metadata()
        
        return self.metadata.tables.get(table_name)
    
    async def close(self):
        """Закриття з'єднань"""
        await self.async_engine.dispose()
        self.sync_engine.dispose()
        logger.info("✅ Database connections closed")


# ============================================================================
# ГЛОБАЛЬНИЙ МЕНЕДЖЕР (для зворотної сумісності)
# ============================================================================

db_manager = DatabaseConnection()


# ============================================================================
# ФУНКЦІЇ ДЛЯ ЗВОРОТНОЇ СУМІСНОСТІ З live_trading.py
# ============================================================================

async def save_trading_signal(db, signal_data: dict) -> int:
    """
    Збереження торгового сигналу в БД
    
    Args:
        db: DatabaseConnection instance
        signal_data: Dict з даними сигналу
        
    Returns:
        ID збереженого сигналу
    """
    try:
        async with db.async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO trading_signals 
                    (symbol, action, confidence, entry_price, stop_loss, take_profit, 
                     quantity, strategy, prediction_source, status, notes, created_at)
                    VALUES 
                    (:symbol, :action, :confidence, :entry_price, :stop_loss, :take_profit,
                     :quantity, :strategy, :prediction_source, :status, :notes, NOW())
                    RETURNING id
                """),
                signal_data
            )
            signal_id = result.scalar_one()
            await session.commit()
            logger.info(f"✅ Signal saved: {signal_data.get('symbol')} {signal_data.get('action')}")
            return signal_id
    except Exception as e:
        logger.error(f"❌ Error saving signal: {e}")
        return None


async def save_position(db, position_data: dict) -> int:
    """
    Збереження позиції в БД
    
    Args:
        db: DatabaseConnection instance
        position_data: Dict з даними позиції
        
    Returns:
        ID збереженої позиції
    """
    try:
        # Конвертуємо metadata в JSON рядок якщо потрібно
        if 'metadata' in position_data and isinstance(position_data['metadata'], dict):
            import json
            position_data['metadata'] = json.dumps(position_data['metadata'])
            
        async with db.async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO positions 
                    (symbol, side, entry_price, quantity, stop_loss, take_profit, 
                     status, signal_id, strategy, entry_time, metadata)
                    VALUES 
                    (:symbol, :side, :entry_price, :quantity, :stop_loss, :take_profit,
                     :status, :signal_id, :strategy, :entry_time, :metadata)
                    RETURNING id
                """),
                position_data
            )
            position_id = result.scalar_one()
            await session.commit()
            logger.info(f"✅ Position saved: {position_data.get('symbol')} {position_data.get('side')}")
            return position_id
    except Exception as e:
        logger.error(f"❌ Error saving position: {e}")
        return None


async def save_trade(db, trade_data: dict) -> int:
    """
    Збереження завершеної угоди в БД
    
    Args:
        db: DatabaseConnection instance
        trade_data: Dict з даними угоди
        
    Returns:
        ID збереженої угоди
    """
    try:
        # Конвертуємо metadata в JSON рядок якщо потрібно
        if 'metadata' in trade_data and isinstance(trade_data['metadata'], dict):
            import json
            trade_data['metadata'] = json.dumps(trade_data['metadata'])
            
        async with db.async_session_factory() as session:
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
                trade_data
            )
            trade_id = result.scalar_one()
            await session.commit()
            
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_percentage', 0)
            logger.info(
                f"✅ Trade saved: {trade_data.get('symbol')} {trade_data.get('side')} "
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
            )
            return trade_id
    except Exception as e:
        logger.error(f"❌ Error saving trade: {e}")
        return None


__all__ = [
    'DatabaseConnection',
    'db_manager',
    'save_trading_signal',
    'save_position',
    'save_trade',
]
