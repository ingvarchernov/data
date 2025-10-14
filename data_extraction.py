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

def get_historical_data(symbol, interval, days_back, api_key=None, api_secret=None, use_public=True):
    try:
        # Якщо use_public=True, не використовуємо API ключі для публічних даних
        exchange_config = {'enableRateLimit': True}
        if not use_public and api_key and api_secret:
            exchange_config.update({
                'apiKey': api_key,
                'secret': api_secret
            })
        
        exchange = ccxt.binance(exchange_config)
        
        # Синхронізація часу з сервером Binance (тільки якщо є ключі)
        if not use_public and api_key and api_secret:
            exchange.load_time_difference()
        
        timeframe = interval
        interval_ms = get_interval_minutes(timeframe) * 60 * 1000
        
        # Binance ліміт - 1000 записів за запит
        # Для days_back днів з інтервалом потрібно завантажувати порціями
        requested_records = days_back * 24 * 60 // get_interval_minutes(timeframe)
        max_per_request = 1000
        
        all_data = []
        since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        while requested_records > 0:
            limit = min(max_per_request, requested_records)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            requested_records -= len(ohlcv)
            
            # Наступний запит починається з часу останньої свічки + 1 інтервал
            since = ohlcv[-1][0] + interval_ms
            
            logger.debug(f"Завантажено {len(ohlcv)} записів, залишилось {requested_records}")
            
            if len(ohlcv) < limit:
                # Отримали менше, ніж запитували - більше даних немає
                break

        data = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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