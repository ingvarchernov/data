"""
Обробка та конвертація даних
"""
import pandas as pd
from typing import List


class DataProcessor:
    """
    Обробник даних
    """
    
    @staticmethod
    def get_interval_ms(interval: str) -> int:
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
    
    @staticmethod
    def process_ohlcv(ohlcv: List[List]) -> pd.DataFrame:
        """Обробка OHLCV даних від CCXT"""
        if not ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Додаткові колонки
        df['quote_av'] = df['volume'] * df['close']
        df['trades'] = 0
        df['tb_base_av'] = df['volume'] * 0.5
        df['tb_quote_av'] = df['quote_av'] * 0.5
        
        return df
    
    @staticmethod
    def klines_to_dataframe(klines: List[List]) -> pd.DataFrame:
        """Конвертація klines від python-binance"""
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Вибір колонок
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Додаткові колонки
        df['quote_av'] = df['volume'] * df['close']
        df['trades'] = 0
        df['tb_base_av'] = df['volume'] * 0.5
        df['tb_quote_av'] = df['quote_av'] * 0.5
        
        return df