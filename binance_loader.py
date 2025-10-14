import logging
import pandas as pd
import os
from dotenv import load_dotenv
from data_extraction import get_historical_data
from sqlalchemy import text

logger = logging.getLogger(__name__)
load_dotenv()

async def save_ohlcv_to_db(db_manager, symbol: str, interval: str, days_back: int = 7):
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    # Використовуємо публічні дані за замовчуванням (без API ключів)
    data = get_historical_data(symbol, interval, days_back, api_key, api_secret, use_public=True)
    if data.empty:
        logger.error(f"❌ Немає даних з Binance для {symbol} {interval}")
        return
    symbol_id = await db_manager.get_or_create_symbol_id(symbol)
    interval_id = await db_manager.get_or_create_interval_id(interval)
    async with db_manager.async_session_factory() as session:
        for _, row in data.iterrows():
            await session.execute(
                text("""
                INSERT INTO historical_data (symbol_id, interval_id, timestamp, open, high, low, close, volume, quote_av, trades, tb_base_av, tb_quote_av)
                VALUES (:symbol_id, :interval_id, :timestamp, :open, :high, :low, :close, :volume, :quote_av, :trades, :tb_base_av, :tb_quote_av)
                ON CONFLICT (symbol_id, interval_id, timestamp) DO NOTHING
                """),
                {
                    "symbol_id": symbol_id,
                    "interval_id": interval_id,
                    "timestamp": row['timestamp'],
                    "open": row['open'],
                    "high": row['high'],
                    "low": row['low'],
                    "close": row['close'],
                    "volume": row['volume'],
                    "quote_av": row['quote_av'],
                    "trades": row['trades'],
                    "tb_base_av": row['tb_base_av'],
                    "tb_quote_av": row['tb_quote_av']
                }
            )
        await session.commit()
    logger.info(f"✅ Завантажено {len(data)} OHLCV для {symbol} {interval}")
