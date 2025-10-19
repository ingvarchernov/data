"""
Збереження даних в базу даних
"""
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DatabaseSaver:
    """
    Зберігач даних в БД
    """
    
    @staticmethod
    async def save_dataframe(
        db_manager,
        symbol: str,
        interval: str,
        data: pd.DataFrame
    ) -> int:
        """
        Збереження DataFrame в БД
        
        Args:
            db_manager: Менеджер БД
            symbol: Символ
            interval: Інтервал
            data: DataFrame з даними
        
        Returns:
            Кількість збережених записів
        """
        if data.empty:
            logger.warning(f"⚠️ Немає даних для збереження {symbol}")
            return 0
        
        try:
            # Отримання ID
            symbol_id = await db_manager.get_or_create_symbol_id(symbol)
            interval_id = await db_manager.get_or_create_interval_id(interval)
            
            # Підготовка записів
            records = DatabaseSaver._prepare_records(
                data, symbol_id, interval_id
            )
            
            # Збереження
            await DatabaseSaver._bulk_insert(db_manager, records)
            
            logger.info(f"✅ Збережено {len(records)} записів для {symbol}")
            return len(records)
        
        except Exception as e:
            logger.error(f"❌ Помилка збереження {symbol}: {e}", exc_info=True)
            return 0
    
    @staticmethod
    def _prepare_records(
        data: pd.DataFrame,
        symbol_id: int,
        interval_id: int
    ) -> list:
        """Підготовка записів для збереження"""
        records = []
        
        for idx, row in data.iterrows():
            records.append({
                'symbol_id': symbol_id,
                'interval_id': interval_id,
                'timestamp': idx if isinstance(idx, datetime) else row.get('timestamp'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'quote_av': float(row.get('quote_av', row['volume'] * row['close'])),
                'trades': int(row.get('trades', 0)),
                'tb_base_av': float(row.get('tb_base_av', row['volume'] * 0.5)),
                'tb_quote_av': float(row.get('tb_quote_av', row.get('quote_av', 0) * 0.5))
            })
        
        return records
    
    @staticmethod
    async def _bulk_insert(db_manager, records: list):
        """Масове збереження записів"""
        async with db_manager.async_session_factory() as session:
            for record in records:
                await session.execute(
                    text("""
                        INSERT INTO historical_data 
                        (symbol_id, interval_id, timestamp, open, high, low, close, 
                         volume, quote_av, trades, tb_base_av, tb_quote_av)
                        VALUES 
                        (:symbol_id, :interval_id, :timestamp, :open, :high, :low, 
                         :close, :volume, :quote_av, :trades, :tb_base_av, :tb_quote_av)
                        ON CONFLICT (symbol_id, interval_id, timestamp)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            quote_av = EXCLUDED.quote_av,
                            trades = EXCLUDED.trades,
                            tb_base_av = EXCLUDED.tb_base_av,
                            tb_quote_av = EXCLUDED.tb_quote_av
                    """),
                    record
                )
            
            await session.commit()