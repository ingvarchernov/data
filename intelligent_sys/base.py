"""
Базові класи та енуми для завантажувачів даних
"""
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class DataSource(Enum):
    """Джерело даних"""
    PYTHON_BINANCE = "python-binance"
    CCXT = "ccxt"
    AUTO = "auto"


@dataclass
class LoaderConfig:
    """Конфігурація завантажувача"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    data_source: DataSource = DataSource.AUTO
    use_public_data: bool = True
    max_retries: int = 3
    retry_delay: int = 1
    rate_limit_delay: float = 0.1
    max_records_per_request: int = 1000
    cache_ttl: int = 300  # 5 хвилин


@dataclass
class DataRange:
    """Діапазон даних для завантаження"""
    symbol: str
    interval: str
    days_back: Optional[int] = None
    start_date: Optional[any] = None
    end_date: Optional[any] = None