"""
Модульна система завантаження даних з Binance та торгових стратегій
"""
import logging
from typing import Optional, List, Dict
import pandas as pd

from .base import DataSource, LoaderConfig, DataRange
from .client_manager import ClientManager
from .data_fetcher import DataFetcher
from .database_saver import DatabaseSaver
from .strategy_integration import StrategyIntegration, create_strategy_integration

logger = logging.getLogger(__name__)


class UnifiedBinanceLoader:
    """
    Об'єднаний завантажувач даних з Binance
    
    Приклад використання:
        loader = UnifiedBinanceLoader(use_public_data=True)
        data = await loader.get_historical_data('BTCUSDT', '1h', days_back=30)
        await loader.close()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        data_source: DataSource = DataSource.AUTO,
        use_public_data: bool = True,
        **kwargs
    ):
        """
        Ініціалізація завантажувача
        
        Args:
            api_key: Binance API ключ
            api_secret: Binance API секрет
            testnet: Використовувати testnet
            data_source: Джерело даних (AUTO, CCXT, PYTHON_BINANCE)
            use_public_data: Публічне API без автентифікації
            **kwargs: Додаткові параметри для LoaderConfig
        """
        # Конфігурація
        self.config = LoaderConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            data_source=data_source,
            use_public_data=use_public_data,
            **kwargs
        )
        
        # Компоненти
        self.client_manager = ClientManager(self.config)
        self.data_fetcher = DataFetcher(self.client_manager)
        self.database_saver = DatabaseSaver()
        
        logger.info(
            f"✅ UnifiedBinanceLoader: source={self.config.data_source.value}, "
            f"testnet={testnet}, public={use_public_data}"
        )
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str = '1h',
        days_back: int = 7,
        start_date: Optional[any] = None,
        end_date: Optional[any] = None
    ) -> pd.DataFrame:
        """Завантаження історичних даних"""
        return await self.data_fetcher.fetch(
            symbol, interval, days_back, start_date, end_date
        )
    
    async def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = '1h',
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """Паралельне завантаження кількох символів"""
        return await self.data_fetcher.fetch_multiple(symbols, interval, days_back)
    
    async def save_to_database(
        self,
        db_manager,
        symbol: str,
        interval: str,
        days_back: int = 7
    ) -> int:
        """Завантаження та збереження в БД"""
        data = await self.get_historical_data(symbol, interval, days_back)
        return await self.database_saver.save_dataframe(
            db_manager, symbol, interval, data
        )
    
    async def close(self):
        """Закриття з'єднань"""
        await self.client_manager.close()
        logger.info("🔒 UnifiedBinanceLoader закрито")


# Експорт
__all__ = [
    'UnifiedBinanceLoader',
    'DataSource',
    'LoaderConfig',
    'ClientManager',
    'DataFetcher',
    'DatabaseSaver',
    'StrategyIntegration',
    'create_strategy_integration'
]