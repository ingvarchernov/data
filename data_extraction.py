# -*- coding: utf-8 -*-
import logging
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from db_utils import insert_symbol, insert_interval, insert_historical_data, get_historical_data_from_db

logger = logging.getLogger(__name__)

def get_historical_data(symbol, interval='1h', days_back=30, api_key=None, api_secret=None):
    """Отримання історичних даних з Binance."""
    try:
        if not api_key or not api_secret:
            logger.error("API ключі (api_key або api_secret) не надані.")
            return pd.DataFrame()

        client = Client(api_key, api_secret)
        interval_map = {
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1M': Client.KLINE_INTERVAL_1MONTH
        }

        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR)
        start_time = (datetime.now() - timedelta(days=days_back)).strftime('%d %b, %Y')

        logger.info(f"Завантаження даних для {symbol} з інтервалом {interval} з Binance.")
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=binance_interval,
            start_str=start_time
        )

        if not klines:
            logger.error(f"Дані для {symbol} за період {start_time} порожні.")
            return pd.DataFrame()

        logger.debug(f"Отримано {len(klines)} записів з Binance API.")
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])

        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']]
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['open'] = data['open'].astype(float)
        data['high'] = data['high'].astype(float)
        data['low'] = data['low'].astype(float)
        data['close'] = data['close'].astype(float)
        data['volume'] = data['volume'].astype(float)
        data['quote_av'] = data['quote_av'].astype(float)
        data['trades'] = data['trades'].astype(int)
        data['tb_base_av'] = data['tb_base_av'].astype(float)
        data['tb_quote_av'] = data['tb_quote_av'].astype(float)

        data = data.drop_duplicates(subset=['timestamp'], keep='last')

        logger.info(f"Завантажено {len(data)} записів для {symbol} з інтервалом {interval}.")
        return data
    except Exception as e:
        logger.error(f"Помилка при завантаженні даних для {symbol}: {e}")
        return pd.DataFrame()