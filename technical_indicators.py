# -*- coding: utf-8 -*-
import logging
import pandas as pd
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
import numpy as np

logger = logging.getLogger(__name__)

def validate_data(data, required_columns):
    """Перевірка наявності колонок і непорожності даних."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Дані повинні бути непорожнім pandas DataFrame.")
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Відсутні колонки: {missing}")

def calculate_moving_average(data, window=20):
    logger.info(f"Розрахунок ковзного середнього з вікном {window}.")
    validate_data(data, ['close'])
    ma = data['close'].rolling(window=window).mean()
    logger.debug(f"MA статистика: mean={ma.mean():.2f}, std={ma.std():.2f}")
    return ma

def calculate_rsi(data, window=14):
    logger.info(f"Розрахунок RSI з вікном {window}.")
    validate_data(data, ['close'])
    rsi = RSIIndicator(data['close'], window=window).rsi()
    logger.debug(f"RSI статистика: mean={rsi.mean():.2f}, std={rsi.std():.2f}")
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    validate_data(data, ['close'])
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line
def calculate_bollinger_bands(data, window=20, num_std=2):
    logger.info(f"Розрахунок Bollinger Bands з вікном {window} і {num_std} стандартними відхиленнями.")
    validate_data(data, ['close'])
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    logger.debug(f"Upper Band mean={upper_band.mean():.2f}, Lower Band mean={lower_band.mean():.2f}")
    return upper_band, lower_band

def calculate_stochastic(data, k_window=14, smooth_k=3, smooth_d=3):
    logger.info(f"Розрахунок Stochastic Oscillator з K={k_window}, smooth_k={smooth_k}, smooth_d={smooth_d}.")
    validate_data(data, ['high', 'low', 'close'])
    stoch = StochasticOscillator(
        high=data['high'], low=data['low'], close=data['close'],
        window=k_window, smooth_window=smooth_d
    )
    return stoch.stoch(), stoch.stoch_signal()

def calculate_ema(data, window=20):
    logger.info(f"Розрахунок EMA з вікном {window}.")
    validate_data(data, ['close'])
    ema = data['close'].ewm(span=window, adjust=False).mean()
    logger.debug(f"EMA статистика: mean={ema.mean():.2f}, std={ema.std():.2f}")
    return ema

def calculate_atr(data, window=14):
    logger.info(f"Розрахунок ATR з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    logger.debug(f"ATR статистика: mean={atr.mean():.2f}, std={atr.std():.2f}")
    return atr

def calculate_cci(data, window=20):
    logger.info(f"Розрахунок CCI з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    mean_price = typical_price.rolling(window=window).mean()
    mean_deviation = np.abs(typical_price - mean_price).rolling(window=window).mean()
    cci = (typical_price - mean_price) / (0.015 * mean_deviation)
    logger.debug(f"CCI статистика: mean={cci.mean():.2f}, std={cci.std():.2f}")
    return cci

def calculate_obv(data):
    logger.info("Розрахунок OBV.")
    validate_data(data, ['close', 'volume'])
    obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
    logger.debug(f"OBV статистика: mean={obv.mean():.2f}, std={obv.std():.2f}")
    return obv

def calculate_adx(data, window=14):
    logger.info(f"Розрахунок ADX з вікном {window}.")
    validate_data(data, ['high', 'low', 'close'])
    adx = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=window)
    adx_value = adx.adx()
    logger.debug(f"ADX статистика: mean={adx_value.mean():.2f}, std={adx_value.std():.2f}")
    return adx_value

def calculate_vwap(data):
    logger.info("Розрахунок VWAP.")
    validate_data(data, ['high', 'low', 'close', 'volume'])
    vwap = VolumeWeightedAveragePrice(
        high=data['high'], low=data['low'], close=data['close'], volume=data['volume']
    ).volume_weighted_average_price()
    logger.debug(f"VWAP статистика: mean={vwap.mean():.2f}, std={vwap.std():.2f}")
    return vwap