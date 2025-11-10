# -*- coding: utf-8 -*-
"""
Об'єднаний завантажувач даних з Binance
Підтримує: історичні дані, real-time streaming, testnet торгівлю
"""
import asyncio
import logging
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import time
from enum import Enum

import pandas as pd
from dotenv import load_dotenv

# Спроба імпорту бібліотек
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception
    BinanceRequestException = Exception

try:
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)
load_dotenv()


class DataSource(Enum):
    """Джерело даних"""
    PYTHON_BINANCE = "python-binance"
    CCXT = "ccxt"
    AUTO = "auto"


class UnifiedBinanceLoader:
    """
    Об'єднаний завантажувач даних з Binance
    
    Особливості:
    - Підтримка різних джерел даних (python-binance, ccxt)
    - Асинхронне завантаження
    - Автоматична пагінація для великих періодів
    - Кешування та retry логіка
    - Підтримка testnet
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        data_source: DataSource = DataSource.AUTO,
        use_public_data: bool = True
    ):
        """
        Ініціалізація завантажувача (SPOT API ONLY)
        
        Args:
            api_key: Binance API ключ
            api_secret: Binance API секрет
            testnet: Використовувати testnet
            data_source: Джерело даних
            use_public_data: Використовувати публічне API (без автентифікації)
        """
        self.testnet = testnet
        self.use_public_data = use_public_data
        
        # Визначення джерела даних
        if data_source == DataSource.AUTO:
            # Для історичних даних завжди використовуємо python-binance, якщо доступно
            if BINANCE_AVAILABLE:
                self.data_source = DataSource.PYTHON_BINANCE
            elif CCXT_AVAILABLE:
                self.data_source = DataSource.CCXT
            else:
                raise ImportError("Потрібно встановити python-binance або ccxt")
        else:
            self.data_source = data_source
        
        # API ключі
        if use_public_data:
            # Для публічних даних використовуємо mainnet ключі якщо testnet=False
            if not testnet and not api_key:
                self.api_key = os.getenv('API_KEY', 'public')
                self.api_secret = os.getenv('API_SECRET', 'public')
                logger.info(f"🔑 Using MAINNET API keys for historical data")
            else:
                self.api_key = "public"
                self.api_secret = "public"
        else:
            # Використовуємо FUTURES_API_KEY / FUTURES_API_SECRET як пріоритетні
            self.api_key = (
                api_key or
                os.getenv('FUTURES_API_KEY') or
                os.getenv('BINANCE_TEST_API_KEY' if testnet else 'API_KEY')
            )
            self.api_secret = (
                api_secret or
                os.getenv('FUTURES_API_SECRET') or
                os.getenv('BINANCE_TEST_API_SECRET' if testnet else 'API_SECRET')
            )
        
        # Ініціалізація клієнтів
        self.binance_client = None
        self.ccxt_client = None
        self._initialize_clients()
        
        # Налаштування
        self.max_retries = 3
        self.retry_delay = 1
        self.rate_limit_delay = 0.1
        self.max_records_per_request = 1000
        
        # Кеш
        self._cache = {}
        self._cache_ttl = 300  # 5 хвилин
        
        logger.info(
            f"✅ UnifiedBinanceLoader ініціалізовано: "
            f"source={self.data_source.value}, testnet={testnet}, "
            f"public={use_public_data}, SPOT API only"
        )

    def _initialize_clients(self):
        """Ініціалізація API клієнтів"""
        try:
            if self.data_source == DataSource.PYTHON_BINANCE:
                if not BINANCE_AVAILABLE:
                    raise ImportError("python-binance не встановлено")
                
                # Якщо маємо реальні ключі (не "public"), використовуємо їх
                if self.api_key and self.api_key != "public":
                    self.binance_client = Client(
                        self.api_key,
                        self.api_secret,
                        testnet=self.testnet
                    )
                    logger.debug(f"🔑 Binance client with API keys (testnet={self.testnet})")
                else:
                    # Публічний клієнт
                    self.binance_client = Client("", "", testnet=self.testnet)
                    logger.debug(f"🌐 Binance public client (testnet={self.testnet})")
                
            elif self.data_source == DataSource.CCXT:
                if not CCXT_AVAILABLE:
                    raise ImportError("ccxt не встановлено")
                
                config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                
                # Testnet URL тільки для authenticated запитів
                if self.testnet and not self.use_public_data:
                    config['urls'] = {'api': 'https://testnet.binance.vision'}
                
                if not self.use_public_data:
                    config['apiKey'] = self.api_key
                    config['secret'] = self.api_secret
                
                self.ccxt_client = ccxt_async.binance(config)
        
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації клієнта: {e}")
            raise

    async def get_historical_data(
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
            days_back: Кількість днів назад (якщо не вказано start_date)
            start_date: Початкова дата
            end_date: Кінцева дата
        
        Returns:
            DataFrame з OHLCV даними
        """
        # Перевірка кешу
        cache_key = f"{symbol}_{interval}_{days_back}_{start_date}_{end_date}"
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                logger.debug(f"📦 Повернення даних з кешу для {symbol}")
                return cached_data
        
        # Розрахунок дат
        if not start_date:
            start_date = datetime.now() - timedelta(days=days_back)
        if not end_date:
            end_date = datetime.now()
        
        # Завантаження через обраний API
        try:
            if self.data_source == DataSource.CCXT:
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
        """Завантаження через CCXT (асинхронно)"""
        if not self.ccxt_client:
            raise RuntimeError("CCXT клієнт не ініціалізований")
        
        all_data = []
        interval_ms = self._get_interval_ms(interval)
        since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        retry_count = 0
        
        while since < end_ts and retry_count < self.max_retries:
            try:
                ohlcv = await self.ccxt_client.fetch_ohlcv(
                    symbol,
                    interval,
                    since=since,
                    limit=self.max_records_per_request
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Наступний запит
                since = ohlcv[-1][0] + interval_ms
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                if len(ohlcv) < self.max_records_per_request:
                    break
                
                retry_count = 0  # Скидаємо лічильник при успіху
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"⚠️ Помилка завантаження (спроба {retry_count}): {e}")
                
                if retry_count >= self.max_retries:
                    raise
                
                await asyncio.sleep(self.retry_delay * retry_count)
        
        return self._process_ohlcv_data(all_data)

    async def _fetch_binance(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Завантаження через python-binance"""
        if not self.binance_client:
            raise RuntimeError("Binance клієнт не ініціалізований")
        
        # python-binance синхронний, тому викликаємо в executor
        loop = asyncio.get_event_loop()
        
        # SPOT API з chunked loading для великих періодів
        # Binance Spot API підтримує необмежену кількість свічок через пагінацію
        klines = await loop.run_in_executor(
            None,
            self.binance_client.get_historical_klines,
            symbol,
            interval,
            int(start_date.timestamp() * 1000),
            int(end_date.timestamp() * 1000)
        )
        
        logger.debug(f"📊 Завантажено {len(klines)} свічок для {symbol} через Spot API")
        
        return self._klines_to_dataframe(klines)

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = '1h',
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Паралельне завантаження даних для кількох символів
        
        Args:
            symbols: Список символів
            interval: Інтервал
            days_back: Кількість днів
        
        Returns:
            Словник {symbol: DataFrame}
        """
        tasks = [
            self.get_historical_data(symbol, interval, days_back)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Помилка завантаження {symbol}: {result}")
            elif isinstance(result, pd.DataFrame) and not result.empty:
                data[symbol] = result
        
        logger.info(f"✅ Завантажено {len(data)}/{len(symbols)} символів")
        return data

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Отримання поточної ціни символу"""
        if self.data_source == DataSource.CCXT:
            if not self.ccxt_client:
                raise RuntimeError("CCXT клієнт не ініціалізований")
            await self.ccxt_client.load_markets()
            ccxt_symbol = symbol
            if symbol in self.ccxt_client.markets_by_id:
                ccxt_symbol = self.ccxt_client.markets_by_id[symbol]['symbol']
            elif '/' not in symbol and symbol.endswith('USDT'):
                ccxt_symbol = f"{symbol[:-4]}/USDT"
            return await self.ccxt_client.fetch_ticker(ccxt_symbol)

        if not self.binance_client:
            raise RuntimeError("Binance клієнт не ініціалізований")

        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(
            None,
            lambda: self.binance_client.get_symbol_ticker(symbol=symbol)
        )
        return {
            'symbol': ticker.get('symbol', symbol),
            'last': float(ticker.get('price', 0.0))
        }

    async def save_to_database(
        self,
        db_manager,
        symbol: str,
        interval: str,
        days_back: int = 7
    ) -> int:
        """
        Завантаження та збереження в БД
        
        Args:
            db_manager: Менеджер БД
            symbol: Символ
            interval: Інтервал
            days_back: Кількість днів
        
        Returns:
            Кількість збережених записів
        """
        # Завантаження даних
        data = await self.get_historical_data(symbol, interval, days_back)
        
        if data.empty:
            logger.error(f"❌ Немає даних для збереження {symbol}")
            return 0
        
        # Отримання ID
        symbol_id = await db_manager.get_or_create_symbol_id(symbol)
        interval_id = await db_manager.get_or_create_interval_id(interval)
        
        # Підготовка даних для batch insert
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
        
        # Збереження в БД
        from sqlalchemy import text
        
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
        
        logger.info(f"✅ Збережено {len(records)} записів для {symbol}")
        return len(records)

    def _get_interval_ms(self, interval: str) -> int:
        """Отримання інтервалу в мілісекундах"""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 60 * 60 * 1000)

    def _process_ohlcv_data(self, ohlcv: List[List]) -> pd.DataFrame:
        """
        Обробка OHLCV даних від CCXT
        ⭐ Мінімізовано апроксимації
        """
        if not ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # ⭐ UTC timezone для консистентності
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # ⭐ Quote asset volume - розраховується як volume * average price
        # Використовуємо (high + low) / 2 як краще наближення ніж просто close
        df['quote_av'] = df['volume'] * ((df['high'] + df['low']) / 2)
        
        # CCXT не надає детальну інформацію про trades
        df['trades'] = 0
        df['tb_base_av'] = df['volume'] * 0.5  # Припущення: 50% від покупців
        df['tb_quote_av'] = df['quote_av'] * 0.5
        
        return df

    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Конвертація klines від python-binance в DataFrame
        ⭐ ВИКОРИСТОВУЄ РЕАЛЬНІ ДАНІ без апроксимацій
        """
        if not klines:
            return pd.DataFrame()
        
        # Binance klines structure:
        # [
        #   0: Open time (ms)
        #   1: Open
        #   2: High
        #   3: Low
        #   4: Close
        #   5: Volume
        #   6: Close time (ms)
        #   7: Quote asset volume
        #   8: Number of trades
        #   9: Taker buy base asset volume
        #   10: Taker buy quote asset volume
        #   11: Ignore
        # ]
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # ⭐ Використовуємо open_time для timestamp (як на Binance UI)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Конвертуємо типи
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'taker_buy_base_asset_volume', 
                       'taker_buy_quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['number_of_trades'] = df['number_of_trades'].astype(int)
        
        # ⭐ ВИКОРИСТОВУЄМО РЕАЛЬНІ ЗНАЧЕННЯ (не апроксимації!)
        df = df.rename(columns={
            'quote_asset_volume': 'quote_av',
            'number_of_trades': 'trades',
            'taker_buy_base_asset_volume': 'tb_base_av',
            'taker_buy_quote_asset_volume': 'tb_quote_av'
        })
        
        # Залишаємо тільки потрібні колонки
        df = df[['open', 'high', 'low', 'close', 'volume', 
                'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']]
        
        return df

    async def close(self):
        """Закриття з'єднань"""
        if self.ccxt_client:
            await self.ccxt_client.close()
        
        logger.info("🔒 UnifiedBinanceLoader закрито")

    def __del__(self):
        """Деструктор"""
        if self.ccxt_client:
            try:
                asyncio.get_event_loop().run_until_complete(self.ccxt_client.close())
            except:
                pass


# Допоміжні функції для зворотної сумісності
async def get_historical_data(
    symbol: str,
    interval: str,
    days_back: int,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    use_public: bool = True
) -> pd.DataFrame:
    """Функція для зворотної сумісності з data_extraction.py"""
    loader = UnifiedBinanceLoader(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        use_public_data=use_public
    )
    
    try:
        return await loader.get_historical_data(symbol, interval, days_back)
    finally:
        await loader.close()


async def save_ohlcv_to_db(
    db_manager,
    symbol: str,
    interval: str,
    days_back: int = 7
):
    """Функція для зворотної сумісності з binance_loader.py"""
    loader = UnifiedBinanceLoader(use_public_data=True)
    
    try:
        return await loader.save_to_database(db_manager, symbol, interval, days_back)
    finally:
        await loader.close()


# Приклад використання
async def example_usage():
    """Приклад використання"""
    
    # 1. Завантаження одного символу
    loader = UnifiedBinanceLoader(use_public_data=True)
    
    btc_data = await loader.get_historical_data('BTCUSDT', '1h', days_back=30)
    print(f"📊 BTC дані: {len(btc_data)} записів")
    print(btc_data.head())
    
    # 2. Завантаження множини символів паралельно
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    multi_data = await loader.get_multiple_symbols(symbols, '1h', 7)
    
    for symbol, data in multi_data.items():
        print(f"✅ {symbol}: {len(data)} записів")
    
    # 3. Збереження в БД (якщо є db_manager)
    # await loader.save_to_database(db_manager, 'BTCUSDT', '1h', 7)
    
    await loader.close()


if __name__ == "__main__":
    asyncio.run(example_usage())