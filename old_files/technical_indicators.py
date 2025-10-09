import logging
import pandas as pd
import numpy as np
import fast_indicators  # Rust-модуль

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
    logger.info(f"Розрахунок RSI з вікном {window} (Rust).")
    validate_data(data, ['close'])
    rsi = fast_indicators.fast_rsi(np.array(data['close'], dtype=np.float64), window)
    # RSI returns len(data) - window values, starting from window position
    result_index = data.index[window:window+len(rsi)]
    return pd.Series(rsi, index=result_index)

def calculate_ema(data, window=20):
    logger.info(f"Розрахунок EMA (Rust).")
    validate_data(data, ['close'])
    ema = fast_indicators.fast_ema(np.array(data['close'], dtype=np.float64), window)
    # EMA returns same length as input
    return pd.Series(ema, index=data.index[:len(ema)])

def calculate_macd(data, fast=12, slow=26, signal=9):
    logger.info(f"Розрахунок MACD (Rust).")
    validate_data(data, ['close'])
    macd, signal_line, histogram = fast_indicators.fast_macd(np.array(data['close'], dtype=np.float64), fast, slow, signal)
    # MACD results start from slow-1 position
    macd_index = data.index[slow-1:slow-1+len(macd)]
    signal_index = data.index[slow-1+signal-1:slow-1+signal-1+len(signal_line)]
    return pd.Series(macd, index=macd_index), pd.Series(signal_line, index=signal_index)

def calculate_bollinger_bands(data, window=20, num_std=2):
    logger.info(f"Розрахунок Bollinger Bands (Rust).")
    validate_data(data, ['close'])
    upper, lower = fast_indicators.fast_bollinger_bands(np.array(data['close'], dtype=np.float64), window, num_std)
    # Bollinger Bands start from window-1 position
    result_index = data.index[window-1:window-1+len(upper)]
    return pd.Series(upper, index=result_index), pd.Series(lower, index=result_index)

def calculate_stochastic(data, k_window=14, smooth_k=3, smooth_d=3):
    logger.info(f"Розрахунок Stochastic Oscillator (Rust).")
    validate_data(data, ['high', 'low', 'close'])
    k, d = fast_indicators.fast_stochastic(
        np.array(data['high'], dtype=np.float64),
        np.array(data['low'], dtype=np.float64),
        np.array(data['close'], dtype=np.float64),
        k_window, smooth_k, smooth_d
    )
    # Stochastic has complex indexing due to multiple smoothing stages
    k_start = k_window - 1 + smooth_k - 1
    d_start = k_start + smooth_d - 1
    k_index = data.index[k_start:k_start+len(k)]
    d_index = data.index[d_start:d_start+len(d)]
    return pd.Series(k, index=k_index), pd.Series(d, index=d_index)

def calculate_atr(data, window=14):
    logger.info(f"Розрахунок ATR (Rust).")
    validate_data(data, ['high', 'low', 'close'])
    atr = fast_indicators.fast_atr(
        np.array(data['high'], dtype=np.float64),
        np.array(data['low'], dtype=np.float64),
        np.array(data['close'], dtype=np.float64),
        window
    )
    # ATR starts from window position (needs 1 previous close for True Range)
    result_index = data.index[window:window+len(atr)]
    return pd.Series(atr, index=result_index)

def calculate_cci(data, window=20):
    logger.info(f"Розрахунок CCI (Rust).")
    validate_data(data, ['high', 'low', 'close'])
    cci = fast_indicators.fast_cci(
        np.array(data['high'], dtype=np.float64),
        np.array(data['low'], dtype=np.float64),
        np.array(data['close'], dtype=np.float64),
        window
    )
    # CCI starts from window-1 position
    result_index = data.index[window-1:window-1+len(cci)]
    return pd.Series(cci, index=result_index)

def calculate_obv(data):
    logger.info("Розрахунок OBV (Rust).")
    validate_data(data, ['close', 'volume'])
    obv = fast_indicators.fast_obv(
        np.array(data['close'], dtype=np.float64),
        np.array(data['volume'], dtype=np.float64)
    )
    # OBV returns same length as input
    return pd.Series(obv, index=data.index[:len(obv)])

def calculate_adx(data, window=14):
    logger.info(f"Розрахунок ADX (Rust).")
    validate_data(data, ['high', 'low', 'close'])
    adx = fast_indicators.fast_adx(
        np.array(data['high'], dtype=np.float64),
        np.array(data['low'], dtype=np.float64),
        np.array(data['close'], dtype=np.float64),
        window
    )
    # ADX has complex calculation requiring 2*window-1 initial values
    result_index = data.index[window*2-1:window*2-1+len(adx)]
    return pd.Series(adx, index=result_index)

def calculate_vwap(data):
    logger.info("Розрахунок VWAP (Rust).")
    validate_data(data, ['high', 'low', 'close', 'volume'])
    vwap = fast_indicators.fast_vwap(
        np.array(data['high'], dtype=np.float64),
        np.array(data['low'], dtype=np.float64),
        np.array(data['close'], dtype=np.float64),
        np.array(data['volume'], dtype=np.float64)
    )
    # VWAP returns same length as input
    return pd.Series(vwap, index=data.index[:len(vwap)])
