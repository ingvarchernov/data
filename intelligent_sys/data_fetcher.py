"""
Завантаження історичних даних з Binance
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd

from .base import DataSource, DataRange
from .client_manager import ClientManager
from .data_processor import DataProcessor

try:  # Опційний імпорт для специфічної обробки помилок CCXT
    from ccxt.base.errors import AuthenticationError as CCXTAuthenticationError  # type: ignore
except Exception:  # pragma: no cover - ccxt може бути відсутній у середовищі
    CCXTAuthenticationError = None

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Завантажувач історичних даних
    """
    
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.config = client_manager.config
        self.processor = DataProcessor()
        
        # Кеш
        self._cache = {}
    
    async def fetch(
        self,
        symbol: str,
        interval: str = '1h',
        days_back: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Завантаження історичних даних
        
        Args:
            symbol: Торговий символ (BTCUSDT)
            interval: Інтервал ('1m', '5m', '15m', '1h', '4h', '1d')
            days_back: Кількість днів назад
            start_date: Початкова дата
            end_date: Кінцева дата
        
        Returns:
            DataFrame з OHLCV даними
        """
        # Перевірка кешу
        cache_key = f"{symbol}_{interval}_{days_back}_{start_date}_{end_date}"
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl:
                logger.debug(f"📦 Кеш для {symbol}")
                return cached_data
        
        # Розрахунок дат
        if not start_date:
            start_date = datetime.now() - timedelta(days=days_back)
        if not end_date:
            end_date = datetime.now()
        
        # Завантаження
        try:
            if self.config.data_source == DataSource.CCXT:
                data = await self._fetch_ccxt(symbol, interval, start_date, end_date)
            else:
                data = await self._fetch_binance(symbol, interval, start_date, end_date)
            
            # Кешування
            if not data.empty:
                self._cache[cache_key] = (data, datetime.now())
            
            logger.info(f"✅ Завантажено {len(data)} записів для {symbol} ({interval})")
            return data
        
        except Exception as e:
            logger.error(f"❌ Помилка завантаження {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def _fetch_ccxt(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Завантаження через CCXT"""
        all_data = []
        interval_ms = self.processor.get_interval_ms(interval)
        since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        retry_count = 0
        
        while since < end_ts and retry_count < self.config.max_retries:
            try:
                client = self.client_manager.get_client()
                if not client:
                    raise RuntimeError("CCXT клієнт не ініціалізований")

                ohlcv = await client.fetch_ohlcv(
                    symbol,
                    interval,
                    since=since,
                    limit=self.config.max_records_per_request
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Наступний запит
                since = ohlcv[-1][0] + interval_ms
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
                
                if len(ohlcv) < self.config.max_records_per_request:
                    break
                
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"⚠️ Помилка (спроба {retry_count}): {e}")

                is_auth_error = (
                    (CCXTAuthenticationError and isinstance(e, CCXTAuthenticationError)) or
                    'Invalid Api-Key ID' in str(e)
                )

                if is_auth_error and not self.config.use_public_data:
                    await self.client_manager.enable_public_data(
                        "Невірний Binance API ключ"
                    )
                    # Після переключення пробуємо повторно з початку
                    return await self._fetch_ccxt(symbol, interval, start_date, end_date)

                if retry_count >= self.config.max_retries:
                    raise
                
                await asyncio.sleep(self.config.retry_delay * retry_count)
        
        return self.processor.process_ohlcv(all_data)
    
    async def _fetch_binance(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Завантаження через python-binance"""
        client = self.client_manager.get_client()
        if not client:
            raise RuntimeError("Binance клієнт не ініціалізований")
        
        loop = asyncio.get_event_loop()
        
        klines = await loop.run_in_executor(
            None,
            client.get_historical_klines,
            symbol,
            interval,
            int(start_date.timestamp() * 1000),
            int(end_date.timestamp() * 1000)
        )
        
        return self.processor.klines_to_dataframe(klines)
    
    async def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = '1h',
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Паралельне завантаження кількох символів
        """
        tasks = [
            self.fetch(symbol, interval, days_back)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"❌ {symbol}: {result}")
            elif isinstance(result, pd.DataFrame) and not result.empty:
                data[symbol] = result
        
        logger.info(f"✅ Завантажено {len(data)}/{len(symbols)} символів")
        return data