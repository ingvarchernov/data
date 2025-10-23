"""
Data Loader - Завантаження даних для тренування

Wrapper для UnifiedBinanceLoader з додатковими функціями:
- Кешування
- Валідація даних
- Автоматична обробка помилок
"""

import asyncio
import logging
import pandas as pd
from typing import Optional
from pathlib import Path
import sys

# Додаємо батьківську директорію
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_binance_loader import UnifiedBinanceLoader

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Завантажувач даних для тренування моделей
    
    Використання:
        loader = DataLoader()
        df = await loader.load('BTCUSDT', '1h', days=365)
        await loader.close()
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = False
    ):
        """
        Args:
            cache_dir: Директорія для кешування
            use_cache: Використовувати кеш
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.loader = None
        
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def load(
        self,
        symbol: str,
        interval: str = '1h',
        days: int = 365,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Завантаження даних
        
        Args:
            symbol: Символ (BTCUSDT)
            interval: Інтервал (1h, 4h, 1d)
            days: Кількість днів
            force_reload: Примусове завантаження (ігнорувати кеш)
        
        Returns:
            DataFrame з OHLCV даними
        """
        # Перевірка кешу
        if self.use_cache and not force_reload:
            cached_data = self._load_from_cache(symbol, interval, days)
            if cached_data is not None:
                logger.info(f"✅ Дані завантажено з кешу: {symbol} {interval}")
                return cached_data
        
        # Завантаження з Binance
        logger.info(f"📊 Завантаження з Binance: {symbol} {interval}, {days} днів")
        
        if self.loader is None:
            self.loader = UnifiedBinanceLoader(use_public_data=True)
        
        df = await self.loader.get_historical_data(
            symbol=symbol,
            interval=interval,
            days_back=days
        )
        
        # Валідація
        if df is None or len(df) == 0:
            raise ValueError(f"Не вдалося завантажити дані для {symbol}")
        
        # Перевірка обов'язкових колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Відсутні колонки: {missing}")
        
        # Збереження в кеш
        if self.use_cache:
            self._save_to_cache(df, symbol, interval, days)
        
        logger.info(f"✅ Завантажено {len(df)} записів")
        return df
    
    def _get_cache_path(self, symbol: str, interval: str, days: int) -> Path:
        """Шлях до кеш файлу"""
        filename = f"{symbol}_{interval}_{days}d.parquet"
        return self.cache_dir / filename
    
    def _load_from_cache(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Завантаження з кешу"""
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(symbol, interval, days)
        
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                logger.warning(f"⚠️ Помилка читання кешу: {e}")
                return None
        
        return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        days: int
    ):
        """Збереження в кеш"""
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(symbol, interval, days)
        
        try:
            df.to_parquet(cache_path)
            logger.info(f"💾 Дані збережено в кеш: {cache_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Помилка збереження кешу: {e}")
    
    def clear_cache(self):
        """Очистити кеш"""
        if self.cache_dir and self.cache_dir.exists():
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
            logger.info("🗑️ Кеш очищено")
    
    async def close(self):
        """Закрити з'єднання"""
        if self.loader:
            await self.loader.close()
            self.loader = None


# Допоміжна синхронна функція
def load_data_sync(
    symbol: str,
    interval: str = '1h',
    days: int = 365,
    **kwargs
) -> pd.DataFrame:
    """
    Синхронна обгортка для load()
    
    Використання:
        df = load_data_sync('BTCUSDT', '1h', days=365)
    """
    async def _load():
        loader = DataLoader(**kwargs)
        try:
            return await loader.load(symbol, interval, days)
        finally:
            await loader.close()
    
    return asyncio.run(_load())
