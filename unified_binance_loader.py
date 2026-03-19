# -*- coding: utf-8 -*-
"""
High-performance unified market data loader.

Goals:
- Keep backward-compatible API for current project.
- Optimize Binance path for speed.
- Provide adapter architecture for future exchanges/asset classes.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    Client = None
    BINANCE_AVAILABLE = False

try:
    import importlib
    ccxt_async = importlib.import_module("ccxt.async_support")
    CCXT_AVAILABLE = True
except ImportError:
    ccxt_async = None
    CCXT_AVAILABLE = False

try:
    from config import BINANCE_CONFIG, MARKET_DATA_CONFIG
except Exception:
    BINANCE_CONFIG = {}
    MARKET_DATA_CONFIG = {}


logger = logging.getLogger(__name__)
load_dotenv()


class DataSource(Enum):
    PYTHON_BINANCE = "python-binance"
    CCXT = "ccxt"
    AUTO = "auto"


class AssetClass(Enum):
    CRYPTO = "crypto"
    FX = "fx"
    EQUITY = "equity"
    COMMODITY = "commodity"


@dataclass
class LoaderSettings:
    max_retries: int = 3
    retry_delay_sec: float = 0.5
    rate_limit_delay_sec: float = 0.0
    max_records_per_request: int = 1000
    cache_ttl_sec: int = 300
    max_parallel_symbols: int = 8


class BaseExchangeAdapter:
    def __init__(
        self,
        exchange_id: str,
        testnet: bool,
        market_type: str,
        api_key: Optional[str],
        api_secret: Optional[str],
        use_public_data: bool,
        settings: LoaderSettings,
    ):
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.market_type = market_type
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_public_data = use_public_data
        self.settings = settings

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class BinanceNativeAdapter(BaseExchangeAdapter):
    """Fast path for Binance using python-binance sync client through executor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance is not installed")

        key = self.api_key or ""
        secret = self.api_secret or ""
        self.client = Client(key, secret, testnet=self.testnet)

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        klines = await loop.run_in_executor(
            None,
            self.client.get_historical_klines,
            symbol,
            interval,
            int(start_date.timestamp() * 1000),
            int(end_date.timestamp() * 1000),
        )

        if not klines:
            return pd.DataFrame()

        return _binance_klines_to_dataframe(klines)

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        ticker = await loop.run_in_executor(
            None,
            lambda: self.client.get_symbol_ticker(symbol=symbol),
        )
        return {
            "symbol": ticker.get("symbol", symbol),
            "last": float(ticker.get("price", 0.0)),
        }


class CcxtAdapter(BaseExchangeAdapter):
    """Universal adapter for many exchanges and future asset classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is not installed")

        exchange_cls = getattr(ccxt_async, self.exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"CCXT exchange '{self.exchange_id}' is not available")

        cfg: Dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": self.market_type},
        }

        if not self.use_public_data and self.api_key and self.api_secret:
            cfg["apiKey"] = self.api_key
            cfg["secret"] = self.api_secret

        self.client = exchange_cls(cfg)

        # Use exchange sandbox mode where supported.
        if self.testnet and hasattr(self.client, "set_sandbox_mode"):
            try:
                self.client.set_sandbox_mode(True)
            except Exception:
                pass

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        symbol_norm = await self._normalize_symbol(symbol)

        all_rows: List[List[Any]] = []
        since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        interval_ms = _interval_to_ms(interval)

        retries = 0
        while since < end_ts:
            try:
                rows = await self.client.fetch_ohlcv(
                    symbol_norm,
                    timeframe=interval,
                    since=since,
                    limit=self.settings.max_records_per_request,
                )
                if not rows:
                    break

                all_rows.extend(rows)
                since = int(rows[-1][0]) + interval_ms

                if len(rows) < self.settings.max_records_per_request:
                    break

                if self.settings.rate_limit_delay_sec > 0:
                    await asyncio.sleep(self.settings.rate_limit_delay_sec)
                retries = 0
            except Exception:
                retries += 1
                if retries > self.settings.max_retries:
                    raise
                await asyncio.sleep(self.settings.retry_delay_sec * retries)

        if not all_rows:
            return pd.DataFrame()

        return _ccxt_ohlcv_to_dataframe(all_rows)

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        symbol_norm = await self._normalize_symbol(symbol)
        ticker = await self.client.fetch_ticker(symbol_norm)
        return {
            "symbol": ticker.get("symbol", symbol_norm),
            "last": float(ticker.get("last") or ticker.get("close") or 0.0),
        }

    async def _normalize_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol

        if symbol.endswith("USDT") and len(symbol) > 4:
            return f"{symbol[:-4]}/USDT"

        return symbol

    async def close(self) -> None:
        await self.client.close()


class UnifiedBinanceLoader:
    """
    Backward-compatible loader name with unified architecture.

    It now supports:
    - Fast Binance-native mode (python-binance).
    - Universal CCXT mode for many exchanges.
    - API compatibility with existing project calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        data_source: DataSource = DataSource.AUTO,
        use_public_data: bool = True,
        exchange_id: Optional[str] = None,
        market_type: str = "spot",
        asset_class: AssetClass = AssetClass.CRYPTO,
    ):
        self.testnet = testnet
        self.use_public_data = use_public_data
        self.asset_class = asset_class
        self.exchange_id = exchange_id or MARKET_DATA_CONFIG.get("default_exchange", "binance")
        self.market_type = market_type or MARKET_DATA_CONFIG.get("default_market_type", "spot")

        self.settings = LoaderSettings(
            max_retries=int(MARKET_DATA_CONFIG.get("max_retries", 3)),
            retry_delay_sec=float(MARKET_DATA_CONFIG.get("retry_delay_sec", 0.5)),
            rate_limit_delay_sec=float(MARKET_DATA_CONFIG.get("rate_limit_delay_sec", 0.0)),
            max_records_per_request=int(MARKET_DATA_CONFIG.get("max_records_per_request", 1000)),
            cache_ttl_sec=int(MARKET_DATA_CONFIG.get("cache_ttl_sec", 300)),
            max_parallel_symbols=int(MARKET_DATA_CONFIG.get("max_parallel_symbols", 8)),
        )

        # Resolve credentials with config/env fallback.
        cfg_key = BINANCE_CONFIG.get("api_key")
        cfg_secret = BINANCE_CONFIG.get("api_secret")
        self.api_key = api_key or cfg_key or os.getenv("API_KEY")
        self.api_secret = api_secret or cfg_secret or os.getenv("API_SECRET")

        self.data_source = self._resolve_data_source(data_source)
        self.adapter: BaseExchangeAdapter = self._build_adapter()

        # Cache: key -> (expires_at_monotonic, dataframe)
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

        logger.info(
            "Unified loader initialized: exchange=%s source=%s market=%s testnet=%s public=%s",
            self.exchange_id,
            self.data_source.value,
            self.market_type,
            self.testnet,
            self.use_public_data,
        )

    def _resolve_data_source(self, data_source: DataSource) -> DataSource:
        if data_source != DataSource.AUTO:
            return data_source

        preferred = str(MARKET_DATA_CONFIG.get("data_source", "auto")).lower()
        if preferred == DataSource.PYTHON_BINANCE.value and BINANCE_AVAILABLE:
            return DataSource.PYTHON_BINANCE
        if preferred == DataSource.CCXT.value and CCXT_AVAILABLE:
            return DataSource.CCXT

        if self.exchange_id == "binance" and BINANCE_AVAILABLE:
            return DataSource.PYTHON_BINANCE
        if CCXT_AVAILABLE:
            return DataSource.CCXT

        raise ImportError("Neither python-binance nor ccxt is available")

    def _build_adapter(self) -> BaseExchangeAdapter:
        if self.data_source == DataSource.PYTHON_BINANCE:
            if self.exchange_id != "binance":
                logger.warning("python-binance adapter supports only binance; falling back to ccxt")
                return CcxtAdapter(
                    exchange_id=self.exchange_id,
                    testnet=self.testnet,
                    market_type=self.market_type,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    use_public_data=self.use_public_data,
                    settings=self.settings,
                )
            return BinanceNativeAdapter(
                exchange_id=self.exchange_id,
                testnet=self.testnet,
                market_type=self.market_type,
                api_key=self.api_key,
                api_secret=self.api_secret,
                use_public_data=self.use_public_data,
                settings=self.settings,
            )

        return CcxtAdapter(
            exchange_id=self.exchange_id,
            testnet=self.testnet,
            market_type=self.market_type,
            api_key=self.api_key,
            api_secret=self.api_secret,
            use_public_data=self.use_public_data,
            settings=self.settings,
        )

    async def get_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days_back: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=days_back)
        if end_date is None:
            end_date = datetime.utcnow()

        cache_key = f"{self.exchange_id}:{self.market_type}:{symbol}:{interval}:{start_date.isoformat()}:{end_date.isoformat()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        df = await self.adapter.fetch_ohlcv(symbol, interval, start_date, end_date)
        if df.empty:
            return df

        normalized = self._normalize_ohlcv(df)
        self._cache_set(cache_key, normalized)
        return normalized

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        days_back: int = 7,
    ) -> Dict[str, pd.DataFrame]:
        semaphore = asyncio.Semaphore(self.settings.max_parallel_symbols)

        async def _task(sym: str) -> Tuple[str, pd.DataFrame]:
            async with semaphore:
                try:
                    data = await self.get_historical_data(sym, interval, days_back)
                    return sym, data
                except Exception as exc:
                    logger.error("Failed to load %s: %s", sym, exc)
                    return sym, pd.DataFrame()

        results = await asyncio.gather(*[_task(s) for s in symbols])
        out: Dict[str, pd.DataFrame] = {}
        for sym, data in results:
            if isinstance(data, pd.DataFrame) and not data.empty:
                out[sym] = data

        logger.info("Loaded %d/%d symbols", len(out), len(symbols))
        return out

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await self.adapter.fetch_ticker(symbol)

    async def save_to_database(
        self,
        db_manager,
        symbol: str,
        interval: str,
        days_back: int = 7,
    ) -> int:
        data = await self.get_historical_data(symbol=symbol, interval=interval, days_back=days_back)
        if data.empty:
            logger.error("No data for database save: %s", symbol)
            return 0

        symbol_id = await db_manager.get_or_create_symbol_id(symbol)
        interval_id = await db_manager.get_or_create_interval_id(interval)

        records: List[Dict[str, Any]] = []
        for ts, row in data.iterrows():
            records.append(
                {
                    "symbol_id": symbol_id,
                    "interval_id": interval_id,
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "quote_av": float(row.get("quote_av", row["volume"] * row["close"])),
                    "trades": int(row.get("trades", 0)),
                    "tb_base_av": float(row.get("tb_base_av", row["volume"] * 0.5)),
                    "tb_quote_av": float(row.get("tb_quote_av", row.get("quote_av", 0.0) * 0.5)),
                }
            )

        from sqlalchemy import text

        query = text(
            """
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
            """
        )

        async with db_manager.async_session_factory() as session:
            await session.execute(query, records)
            await session.commit()

        logger.info("Saved %d rows for %s", len(records), symbol)
        return len(records)

    async def close(self) -> None:
        await self.adapter.close()

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        item = self._cache.get(key)
        if item is None:
            return None

        expires_at, df = item
        if time.monotonic() > expires_at:
            self._cache.pop(key, None)
            return None

        return df

    def _cache_set(self, key: str, df: pd.DataFrame) -> None:
        self._cache[key] = (time.monotonic() + self.settings.cache_ttl_sec, df)

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Ensure datetime UTC index.
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.set_index("timestamp")
        elif not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

        out = out[~out.index.isna()]
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]

        # Required OHLCV columns.
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Optional columns normalized for existing pipeline.
        if "quote_av" not in out.columns:
            out["quote_av"] = out["volume"] * out["close"]
        else:
            out["quote_av"] = pd.to_numeric(out["quote_av"], errors="coerce")

        if "trades" not in out.columns:
            out["trades"] = 0
        else:
            out["trades"] = pd.to_numeric(out["trades"], errors="coerce").fillna(0).astype(int)

        if "tb_base_av" not in out.columns:
            out["tb_base_av"] = out["volume"] * 0.5
        else:
            out["tb_base_av"] = pd.to_numeric(out["tb_base_av"], errors="coerce")

        if "tb_quote_av" not in out.columns:
            out["tb_quote_av"] = out["quote_av"] * 0.5
        else:
            out["tb_quote_av"] = pd.to_numeric(out["tb_quote_av"], errors="coerce")

        out = out[["open", "high", "low", "close", "volume", "quote_av", "trades", "tb_base_av", "tb_quote_av"]]
        out = out.dropna(subset=required)

        return out


def _interval_to_ms(interval: str) -> int:
    interval_map = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "3d": 259_200_000,
        "1w": 604_800_000,
        "1M": 2_592_000_000,
    }
    return interval_map.get(interval, 3_600_000)


def _ccxt_ohlcv_to_dataframe(ohlcv: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["quote_av"] = df["volume"] * ((df["high"] + df["low"]) / 2.0)
    df["trades"] = 0
    df["tb_base_av"] = df["volume"] * 0.5
    df["tb_quote_av"] = df["quote_av"] * 0.5
    return df


def _binance_klines_to_dataframe(klines: List[List[Any]]) -> pd.DataFrame:
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    num_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    df[num_cols] = df[num_cols].astype(float)
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)

    out = df.rename(
        columns={
            "quote_asset_volume": "quote_av",
            "number_of_trades": "trades",
            "taker_buy_base_asset_volume": "tb_base_av",
            "taker_buy_quote_asset_volume": "tb_quote_av",
        }
    )
    return out[["open", "high", "low", "close", "volume", "quote_av", "trades", "tb_base_av", "tb_quote_av"]]


# Backward-compatible helper functions.
async def get_historical_data(
    symbol: str,
    interval: str,
    days_back: int,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    use_public: bool = True,
) -> pd.DataFrame:
    loader = UnifiedBinanceLoader(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        use_public_data=use_public,
    )
    try:
        return await loader.get_historical_data(symbol, interval, days_back)
    finally:
        await loader.close()


async def save_ohlcv_to_db(
    db_manager,
    symbol: str,
    interval: str,
    days_back: int = 7,
):
    loader = UnifiedBinanceLoader(use_public_data=True)
    try:
        return await loader.save_to_database(db_manager, symbol, interval, days_back)
    finally:
        await loader.close()


async def example_usage() -> None:
    loader = UnifiedBinanceLoader(use_public_data=True, exchange_id="binance")
    try:
        btc_data = await loader.get_historical_data("BTCUSDT", "1h", days_back=30)
        print(f"BTC rows: {len(btc_data)}")

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        data = await loader.get_multiple_symbols(symbols, "1h", 7)
        for sym, df in data.items():
            print(f"{sym}: {len(df)}")
    finally:
        await loader.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
