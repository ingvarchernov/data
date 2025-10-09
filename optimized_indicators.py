# -*- coding: utf-8 -*-
"""
Оптимізований модуль для розрахунку технічних індикаторів
Використовує RUST як основний движок з мінімальними Python fallback'ами
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

# Спробуємо імпортувати Rust модуль
try:
    import fast_indicators
    RUST_AVAILABLE = True
    logger.info("✅ Rust модуль fast_indicators завантажено")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"⚠️ Rust модуль недоступний: {e}. Використовуватиметься Python fallback")

# Backup функції для випадків, коли Rust недоступний
def _pandas_fallback_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Python fallback для RSI"""
    close = data['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()

def _pandas_fallback_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Python fallback для EMA"""
    return data['close'].ewm(span=period).mean()

def _pandas_fallback_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Python fallback для MACD"""
    ema_fast = data['close'].ewm(span=fast).mean()
    ema_slow = data['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line.dropna(), signal_line.dropna()

def _pandas_fallback_bollinger(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Python fallback для Bollinger Bands"""
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.dropna(), lower.dropna()

def _pandas_fallback_stochastic(data: pd.DataFrame, k_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Python fallback для Stochastic"""
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return k_smooth.dropna(), d_smooth.dropna()

def _pandas_fallback_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Python fallback для Williams %R"""
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    return williams_r.dropna()

def _pandas_fallback_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Python fallback для Commodity Channel Index"""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mad)
    return cci.dropna()

def _pandas_fallback_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Python fallback для Average Directional Index"""
    high = data['high']
    low = data['low']
    close = data['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (pd.Series(dm_plus).rolling(window=period).mean() / atr)
    di_minus = 100 * (pd.Series(dm_minus).rolling(window=period).mean() / atr)

    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx.dropna()

class OptimizedIndicatorCalculator:
    """Оптимізований калькулятор технічних індикаторів з пріоритетом Rust"""
    
    def __init__(self, use_async: bool = True, n_workers: int = 4):
        self.use_async = use_async
        self.n_workers = n_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=n_workers) if use_async else None
        
    def __del__(self):
        """Очистка ресурсів"""
        if hasattr(self, 'thread_executor') and self.thread_executor:
            self.thread_executor.shutdown(wait=False)

    async def calculate_rsi_async(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Асинхронний розрахунок RSI з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_rsi(data, period)
            
        try:
            prices = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                rsi_values = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_rsi,
                    prices, period
                )
            else:
                rsi_values = fast_indicators.fast_rsi(prices, period)
            
            # Створюємо правильний індекс для результату
            result_index = data.index[period:period+len(rsi_values)]
            return pd.Series(rsi_values, index=result_index, name='RSI')
            
        except Exception as e:
            logger.warning(f"Rust RSI failed, fallback to pandas: {e}")
            return _pandas_fallback_rsi(data, period)

    async def calculate_ema_async(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Асинхронний розрахунок EMA з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_ema(data, period)
            
        try:
            prices = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                ema_values = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_ema,
                    prices, period
                )
            else:
                ema_values = fast_indicators.fast_ema(prices, period)
                
            return pd.Series(ema_values, index=data.index[:len(ema_values)], name='EMA')
            
        except Exception as e:
            logger.warning(f"Rust EMA failed, fallback to pandas: {e}")
            return _pandas_fallback_ema(data, period)

    async def calculate_macd_async(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Асинхронний розрахунок MACD з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_macd(data, fast, slow, signal)
            
        try:
            prices = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                macd, signal_line, histogram = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_macd,
                    prices, fast, slow, signal
                )
            else:
                macd, signal_line, histogram = fast_indicators.fast_macd(prices, fast, slow, signal)
                
            macd_index = data.index[slow-1:slow-1+len(macd)]
            signal_index = data.index[slow-1+signal-1:slow-1+signal-1+len(signal_line)]
            return (pd.Series(macd, index=macd_index, name='MACD'), 
                   pd.Series(signal_line, index=signal_index, name='MACD_Signal'))
                   
        except Exception as e:
            logger.warning(f"Rust MACD failed, fallback to pandas: {e}")
            return _pandas_fallback_macd(data, fast, slow, signal)

    async def calculate_bollinger_bands_async(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Асинхронний розрахунок Bollinger Bands з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_bollinger(data, period, std_dev)
            
        try:
            prices = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                upper, lower = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_bollinger_bands,
                    prices, period, std_dev
                )
            else:
                upper, lower = fast_indicators.fast_bollinger_bands(prices, period, std_dev)
                
            result_index = data.index[period-1:period-1+len(upper)]
            return (pd.Series(upper, index=result_index, name='Upper_Band'),
                   pd.Series(lower, index=result_index, name='Lower_Band'))
                   
        except Exception as e:
            logger.warning(f"Rust Bollinger Bands failed, fallback to pandas: {e}")
            return _pandas_fallback_bollinger(data, period, std_dev)

    async def calculate_stochastic_async(self, data: pd.DataFrame, k_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Асинхронний розрахунок Stochastic з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_stochastic(data, k_period, smooth_k, smooth_d)
            
        try:
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            close = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                k, d = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_stochastic,
                    high, low, close, k_period, smooth_k, smooth_d
                )
            else:
                k, d = fast_indicators.fast_stochastic(high, low, close, k_period, smooth_k, smooth_d)
                
            k_start = k_period - 1 + smooth_k - 1
            d_start = k_start + smooth_d - 1
            k_index = data.index[k_start:k_start+len(k)]
            d_index = data.index[d_start:d_start+len(d)]
            return (pd.Series(k, index=k_index, name='Stoch'),
                   pd.Series(d, index=d_index, name='Stoch_Signal'))
                   
        except Exception as e:
            logger.warning(f"Rust Stochastic failed, fallback to pandas: {e}")
            return _pandas_fallback_stochastic(data, k_period, smooth_k, smooth_d)

    async def calculate_atr_async(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Асинхронний розрахунок ATR з пріоритетом Rust"""
        if not RUST_AVAILABLE:
            return _pandas_fallback_atr(data, period)
            
        try:
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            close = data['close'].values.astype(np.float64)
            
            if self.use_async and self.thread_executor:
                loop = asyncio.get_event_loop()
                atr_values = await loop.run_in_executor(
                    self.thread_executor,
                    fast_indicators.fast_atr,
                    high, low, close, period
                )
            else:
                atr_values = fast_indicators.fast_atr(high, low, close, period)
                
            result_index = data.index[period:period+len(atr_values)]
            return pd.Series(atr_values, index=result_index, name='ATR')
            
        except Exception as e:
            logger.warning(f"Rust ATR failed, fallback to pandas: {e}")
            return _pandas_fallback_atr(data, period)

    # Додаткові індикатори які можуть бути відсутні в Rust
    async def calculate_cci_async(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Розрахунок CCI"""
        if RUST_AVAILABLE:
            try:
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                close = data['close'].values.astype(np.float64)
                
                if self.use_async and self.thread_executor:
                    loop = asyncio.get_event_loop()
                    cci_values = await loop.run_in_executor(
                        self.thread_executor,
                        fast_indicators.fast_cci,
                        high, low, close, period
                    )
                else:
                    cci_values = fast_indicators.fast_cci(high, low, close, period)
                    
                result_index = data.index[period-1:period-1+len(cci_values)]
                return pd.Series(cci_values, index=result_index, name='CCI')
                
            except Exception as e:
                logger.warning(f"Rust CCI failed, fallback to pandas: {e}")
        
        # Pandas fallback
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci.dropna()

    async def calculate_williams_r_async(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Асинхронний розрахунок Williams %R"""
        # Простий pandas fallback (Williams %R відносно простий)
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        result = williams_r.dropna()
        result.name = 'Williams_R'
        return result

    async def calculate_cci_async(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Асинхронний розрахунок CCI"""
        if RUST_AVAILABLE:
            try:
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                close = data['close'].values.astype(np.float64)
                
                if self.use_async and self.thread_executor:
                    loop = asyncio.get_event_loop()
                    cci_values = await loop.run_in_executor(
                        self.thread_executor,
                        fast_indicators.fast_cci,
                        high, low, close, period
                    )
                else:
                    cci_values = fast_indicators.fast_cci(high, low, close, period)
                
                result_index = data.index[period-1:period-1+len(cci_values)]
                return pd.Series(cci_values, index=result_index, name='CCI')
                
            except Exception as e:
                logger.warning(f"Rust CCI failed, fallback to pandas: {e}")
        
        # Pandas fallback
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci.dropna()

    async def calculate_adx_async(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Розрахунок ADX"""
        # Простий pandas fallback (ADX досить складний для реалізації)
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        di_plus = (dm_plus.rolling(window=period).mean() / atr) * 100
        di_minus = (dm_minus.rolling(window=period).mean() / atr) * 100
        
        # ADX calculation
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx.dropna()

    async def calculate_vwap_async(self, data: pd.DataFrame) -> pd.Series:
        """Розрахунок VWAP"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap

    async def calculate_all_indicators_batch(self, data: pd.DataFrame, config: Dict = None) -> Dict[str, pd.Series]:
        """Пакетний розрахунок всіх індикаторів"""
        if config is None:
            config = {
                'rsi_period': 14,
                'ema_period': 20,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2.0,
                'stoch_k': 14,
                'stoch_smooth_k': 3,
                'stoch_smooth_d': 3,
                'atr_period': 14
            }
        
        # Запускаємо всі розрахунки паралельно
        tasks = [
            self.calculate_rsi_async(data, config['rsi_period']),
            self.calculate_ema_async(data, config['ema_period']),
            self.calculate_macd_async(data, config['macd_fast'], config['macd_slow'], config['macd_signal']),
            self.calculate_bollinger_bands_async(data, config['bb_period'], config['bb_std']),
            self.calculate_stochastic_async(data, config['stoch_k'], config['stoch_smooth_k'], config['stoch_smooth_d']),
            self.calculate_atr_async(data, config['atr_period']),
            # Нові індикатори
            self.calculate_williams_r_async(data, 14),
            self.calculate_cci_async(data, 20),
            self.calculate_ema_async(data, 10),  # EMA_10
            self.calculate_ema_async(data, 50),  # EMA_50
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Розпаковуємо результати
        rsi = results[0]
        ema_20 = results[1]
        macd, macd_signal = results[2]
        bb_upper, bb_lower = results[3]
        stoch_k, stoch_d = results[4]
        atr = results[5]
        williams_r = results[6]
        cci = results[7]
        ema_10 = results[8]
        ema_50 = results[9]
        
        return {
            'RSI': rsi,
            'EMA_20': ema_20,
            'EMA_10': ema_10,
            'EMA_50': ema_50,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'BB_Upper': bb_upper,
            'BB_Lower': bb_lower,
            'Stoch_K': stoch_k,
            'Stoch_D': stoch_d,
            'ATR': atr,
            'Williams_R': williams_r,
            'CCI': cci
        }

# ============================================================================
# FALLBACK ФУНКЦІЇ (коли Rust недоступний)
# ============================================================================

def _pandas_fallback_rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """Pandas fallback для RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()

def _pandas_fallback_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Pandas fallback для EMA"""
    return data['close'].ewm(span=period).mean()

def _pandas_fallback_macd(data: pd.DataFrame, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    """Pandas fallback для MACD"""
    ema_fast = data['close'].ewm(span=fast).mean()
    ema_slow = data['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd.dropna(), signal_line.dropna()

def _pandas_fallback_bollinger(data: pd.DataFrame, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series]:
    """Pandas fallback для Bollinger Bands"""
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.dropna(), lower.dropna()

def _pandas_fallback_stochastic(data: pd.DataFrame, k_period: int, smooth_k: int, smooth_d: int) -> Tuple[pd.Series, pd.Series]:
    """Pandas fallback для Stochastic"""
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()
    return k_smooth.dropna(), d_smooth.dropna()

def _pandas_fallback_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Pandas fallback для ATR"""
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.dropna()

# ============================================================================
# ПУБЛІЧНІ ФУНКЦІЇ ДЛЯ ЗВОРОТНОЇ СУМІСНОСТІ
# ============================================================================

# Глобальний калькулятор для використання в інших модулях
global_calculator = OptimizedIndicatorCalculator()

# Синхронні обгортки для зворотної сумісності
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Синхронна обгортка для RSI"""
    return asyncio.run(global_calculator.calculate_rsi_async(data, window))

def calculate_ema(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Синхронна обгортка для EMA"""
    return asyncio.run(global_calculator.calculate_ema_async(data, window))

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Синхронна обгортка для MACD"""
    return asyncio.run(global_calculator.calculate_macd_async(data, fast, slow, signal))

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Синхронна обгортка для Bollinger Bands"""
    return asyncio.run(global_calculator.calculate_bollinger_bands_async(data, window, num_std))

def calculate_stochastic(data: pd.DataFrame, k_window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Синхронна обгортка для Stochastic"""
    return asyncio.run(global_calculator.calculate_stochastic_async(data, k_window, smooth_k, smooth_d))

def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Синхронна обгортка для ATR"""
    return asyncio.run(global_calculator.calculate_atr_async(data, window))

def calculate_all_indicators(data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Розрахунок всіх технічних індикаторів
    
    Args:
        data: OHLCV данные
        config: Конфігурація індикаторів
    
    Returns:
        DataFrame з усіма індикаторами
    """
    if config is None:
        config = {
            'rsi_period': 14,
            'ema_periods': [10, 20, 50],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'stoch_k': 14,
            'stoch_smooth_k': 3,
            'stoch_smooth_d': 3,
            'atr_period': 14
        }
    
    calculator = OptimizedIndicatorCalculator()
    result = data.copy()
    
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # RSI
        result['RSI'] = loop.run_until_complete(
            calculator.calculate_rsi_async(data, config['rsi_period'])
        )
        
        # EMA
        for period in config['ema_periods']:
            result[f'EMA_{period}'] = loop.run_until_complete(
                calculator.calculate_ema_async(data, period)
            )
        
        # MACD
        macd, macd_signal = loop.run_until_complete(
            calculator.calculate_macd_async(
                data, config['macd_fast'], config['macd_slow'], config['macd_signal']
            )
        )
        result['MACD'] = macd
        result['MACD_Signal'] = macd_signal
        
        # Bollinger Bands
        bb_upper, bb_lower = loop.run_until_complete(
            calculator.calculate_bollinger_bands_async(
                data, config['bb_period'], config['bb_std']
            )
        )
        result['BB_Upper'] = bb_upper
        result['BB_Lower'] = bb_lower
        
        # Stochastic
        stoch_k, stoch_d = loop.run_until_complete(
            calculator.calculate_stochastic_async(
                data, config['stoch_k'], config['stoch_smooth_k'], config['stoch_smooth_d']
            )
        )
        result['Stoch_K'] = stoch_k
        result['Stoch_D'] = stoch_d
        
        # ATR
        result['ATR'] = loop.run_until_complete(
            calculator.calculate_atr_async(data, config['atr_period'])
        )
        
        # Додаткові індикатори
        result['CCI'] = loop.run_until_complete(
            calculator.calculate_cci_async(data)
        )
        result['OBV'] = loop.run_until_complete(
            calculator.calculate_obv_async(data)
        )
        result['ADX'] = loop.run_until_complete(
            calculator.calculate_adx_async(data)
        )
        result['VWAP'] = loop.run_until_complete(
            calculator.calculate_vwap_async(data)
        )
        
        loop.close()
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Fallback до базових pandas розрахунків
        result = _fallback_all_indicators(data, config)
    
    return result

def _fallback_all_indicators(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fallback розрахунок всіх індикаторів через pandas"""
    result = data.copy()
    
    # RSI
    result['RSI'] = _pandas_fallback_rsi(data, config['rsi_period'])
    
    # EMA
    for period in config['ema_periods']:
        result[f'EMA_{period}'] = data['close'].ewm(span=period).mean()
    
    # MACD
    macd, macd_signal = _pandas_fallback_macd(data, config['macd_fast'], config['macd_slow'], config['macd_signal'])
    result['MACD'] = macd
    result['MACD_Signal'] = macd_signal
    
    # Bollinger Bands
    bb_upper, bb_lower = _pandas_fallback_bollinger(data, config['bb_period'], config['bb_std'])
    result['BB_Upper'] = bb_upper
    result['BB_Lower'] = bb_lower
    
    # Stochastic
    stoch_k, stoch_d = _pandas_fallback_stochastic(data, config['stoch_k'], config['stoch_smooth_k'], config['stoch_smooth_d'])
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d
    
    # ATR
    result['ATR'] = _pandas_fallback_atr(data, config['atr_period'])
    
    return result

def get_performance_stats() -> dict:
    """Отримання статистики продуктивності"""
    return {
        'rust_available': RUST_AVAILABLE,
        'rust_module_path': 'fast_indicators/target/release/libfast_indicators.so' if RUST_AVAILABLE else None,
        'fallback_method': 'pandas' if not RUST_AVAILABLE else None,
        'expected_speedup': '25x' if RUST_AVAILABLE else '1x (baseline)',
        'recommended_batch_size': 1000 if RUST_AVAILABLE else 100
    }