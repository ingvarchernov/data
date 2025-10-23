"""
Модульна система завантаження даних з Binance та торгових стратегій

Цей модуль надає компонентну архітектуру для:
- Завантаження даних з Binance (ClientManager, DataFetcher)
- Збереження в БД (DatabaseSaver)
- Інтеграція стратегій (StrategyIntegration)
- Об'єднаний loader (UnifiedBinanceLoader з головного модуля)
"""
import sys
from pathlib import Path

# Додаємо батьківську директорію для імпорту UnifiedBinanceLoader
sys.path.insert(0, str(Path(__file__).parent.parent))

# Імпорт компонентів з модулів
from .base import DataSource, LoaderConfig, DataRange
from .client_manager import ClientManager
from .data_fetcher import DataFetcher
from .database_saver import DatabaseSaver
from .strategy_integration import StrategyIntegration, create_strategy_integration

# Імпорт головного UnifiedBinanceLoader (уникаємо дублювання)
try:
    from unified_binance_loader import UnifiedBinanceLoader
except ImportError:
    # Fallback - створюємо wrapper якщо основний файл недоступний
    import logging
    from typing import Optional, List, Dict
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    
    class UnifiedBinanceLoader:
        """
        Wrapper для модульної архітектури
        Використовує компоненти з intelligent_sys
        """
        
        def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = True,
            data_source: DataSource = DataSource.AUTO,
            use_public_data: bool = True,
            public_only: bool = True,  # Альтернативна назва
            **kwargs
        ):
            """Ініціалізація компонентного завантажувача"""
            # public_only - синонім use_public_data
            if public_only:
                use_public_data = True
                
            self.config = LoaderConfig(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                data_source=data_source,
                use_public_data=use_public_data,
                **kwargs
            )
            
            self.client_manager = ClientManager(self.config)
            self.data_fetcher = DataFetcher(self.client_manager)
            self.database_saver = DatabaseSaver()
            
            logger.info(
                f"✅ UnifiedBinanceLoader (modular): source={self.config.data_source.value}, "
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
        
        # Alias для сумісності
        async def load_data_async(self, symbol: str, interval: str, days: int = 7) -> pd.DataFrame:
            """Alias для get_historical_data"""
            return await self.get_historical_data(symbol, interval, days_back=days)
        
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
            logger.info("🔒 UnifiedBinanceLoader (modular) закрито")


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