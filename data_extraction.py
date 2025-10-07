import pandas as pd
import logging
from datetime import datetime, timedelta
import ccxt
import sys
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

def get_historical_data(symbol, interval, days_back, api_key, api_secret):
    try:
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        timeframe = interval
        limit = days_back * 24 * 60 // get_interval_minutes(timeframe)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['quote_av'] = data['volume'] * data['close']
        data['trades'] = 0
        data['tb_base_av'] = data['volume'] * 0.5
        data['tb_quote_av'] = data['quote_av'] * 0.5

        logger.info(f"Завантажено {len(data)} записів для {symbol} з інтервалом {interval}.")
        logger.debug(f"Діапазон даних: від {data['timestamp'].min()} до {data['timestamp'].max()}")
        return data
    except Exception as e:
        logger.error(f"Помилка при завантаженні даних для {symbol} ({interval}): {e}")
        return pd.DataFrame()

def get_interval_minutes(interval):
    interval_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '1w': 7*24*60, '1M': 30*24*60}
    return interval_map.get(interval, 60)