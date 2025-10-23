"""
Optimized Indicator Calculator

Головний клас для розрахунку всіх індикаторів з підтримкою:
- Асинхронного виконання
- Rust acceleration
- Batch processing
- Automatic fallback до pandas
"""

import logging
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from . import RUST_AVAILABLE
from .trend import (
    calculate_sma_async,
    calculate_ema_async,
    calculate_macd_async,
    calculate_trix_async
)
from .momentum import (
    calculate_rsi_async,
    calculate_stochastic_async,
    calculate_roc_async,
    calculate_momentum_async,
    calculate_williams_r_async,
    calculate_cci_async
)
from .volatility import (
    calculate_atr_async,
    calculate_bollinger_bands_async,
    calculate_adx_async
)
from .volume import (
    calculate_obv_async,
    calculate_vwap_async,
    calculate_volume_sma_async,
    calculate_volume_ratio_async
)

logger = logging.getLogger(__name__)


class OptimizedIndicatorCalculator:
    """
    Оптимізований калькулятор технічних індикаторів
    
    Features:
    - Асинхронний batch розрахунок
    - Rust acceleration (якщо доступний)
    - ThreadPoolExecutor для паралелізації
    - Автоматичний fallback до pandas
    
    Example:
        calculator = OptimizedIndicatorCalculator(use_async=True, n_workers=4)
        indicators = await calculator.calculate_all_indicators_batch(df, config)
    """
    
    def __init__(self, use_async: bool = True, n_workers: int = 4):
        """
        Ініціалізація калькулятора
        
        Args:
            use_async: Використовувати асинхронність
            n_workers: Кількість worker threads
        """
        self.use_async = use_async
        self.n_workers = n_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=n_workers) if use_async else None
        
        logger.info(
            f"📊 OptimizedIndicatorCalculator initialized: "
            f"async={use_async}, workers={n_workers}, rust={RUST_AVAILABLE}"
        )
    
    def __del__(self):
        """Очистка ресурсів"""
        if hasattr(self, 'thread_executor') and self.thread_executor:
            self.thread_executor.shutdown(wait=False)
    
    async def calculate_all_indicators_batch(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> Dict[str, pd.Series]:
        """
        Пакетний розрахунок всіх індикаторів
        
        Args:
            data: DataFrame with OHLCV data
            config: Configuration dict with indicator parameters
        
        Returns:
            Dict with all calculated indicators
        """
        if config is None:
            config = self._default_config()
        
        executor = self.thread_executor if self.use_async else None
        
        # Запускаємо всі розрахунки паралельно
        tasks = [
            # Trend
            calculate_rsi_async(data, config['rsi_period'], executor),
            calculate_ema_async(data, config['ema_20'], executor),
            calculate_ema_async(data, config['ema_10'], executor),
            calculate_ema_async(data, config['ema_50'], executor),
            calculate_macd_async(
                data,
                config['macd_fast'],
                config['macd_slow'],
                config['macd_signal'],
                executor
            ),
            calculate_trix_async(data, config.get('trix_period', 15), executor),
            
            # Momentum
            calculate_stochastic_async(
                data,
                config['stoch_k'],
                config['stoch_smooth_k'],
                config['stoch_smooth_d'],
                executor
            ),
            calculate_roc_async(data, config.get('roc_period', 10), executor),
            calculate_momentum_async(data, config.get('momentum_period', 10), executor),
            calculate_williams_r_async(data, config.get('williams_r_period', 14), executor),
            calculate_cci_async(data, config.get('cci_period', 20), executor),
            
            # Volatility
            calculate_atr_async(data, config['atr_period'], executor),
            calculate_bollinger_bands_async(
                data,
                config['bb_period'],
                config['bb_std'],
                executor
            ),
            calculate_adx_async(data, config.get('adx_period', 14), executor),
            
            # Volume
            calculate_obv_async(data, executor),
            calculate_vwap_async(data, executor),
            calculate_volume_sma_async(data, config.get('volume_sma_period', 20), executor),
            calculate_volume_ratio_async(data, config.get('volume_ratio_period', 20), executor),
        ]
        
        logger.info(f"🚀 Запуск {len(tasks)} індикаторів паралельно...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Розпаковуємо результати
        indicators = {}
        
        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {i} failed: {result}")
        
        try:
            idx = 0
            
            # Trend
            indicators['RSI'] = results[idx]; idx += 1
            indicators['EMA_20'] = results[idx]; idx += 1
            indicators['EMA_10'] = results[idx]; idx += 1
            indicators['EMA_50'] = results[idx]; idx += 1
            
            macd, macd_signal, macd_hist = results[idx]; idx += 1
            indicators['MACD'] = macd
            indicators['MACD_Signal'] = macd_signal
            indicators['MACD_Histogram'] = macd_hist
            
            indicators['TRIX'] = results[idx]; idx += 1
            
            # Momentum
            stoch_k, stoch_d = results[idx]; idx += 1
            indicators['Stoch_K'] = stoch_k
            indicators['Stoch_D'] = stoch_d
            
            indicators['ROC'] = results[idx]; idx += 1
            indicators['Momentum'] = results[idx]; idx += 1
            indicators['Williams_R'] = results[idx]; idx += 1
            indicators['CCI'] = results[idx]; idx += 1
            
            # Volatility
            indicators['ATR'] = results[idx]; idx += 1
            
            bb_upper, bb_middle, bb_lower = results[idx]; idx += 1
            indicators['BB_Upper'] = bb_upper
            indicators['BB_Middle'] = bb_middle
            indicators['BB_Lower'] = bb_lower
            
            indicators['ADX'] = results[idx]; idx += 1
            
            # Volume
            indicators['OBV'] = results[idx]; idx += 1
            indicators['VWAP'] = results[idx]; idx += 1
            indicators['Volume_SMA'] = results[idx]; idx += 1
            indicators['Volume_Ratio'] = results[idx]; idx += 1
            
        except Exception as e:
            logger.error(f"❌ Error unpacking results: {e}")
            raise
        
        logger.info(f"✅ Розраховано {len(indicators)} індикаторів")
        
        return indicators
    
    async def calculate_all_to_dataframe(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Розрахунок всіх індикаторів і повернення як DataFrame
        
        Args:
            data: DataFrame with OHLCV data
            config: Configuration dict
        
        Returns:
            DataFrame with all indicators added
        """
        indicators = await self.calculate_all_indicators_batch(data, config)
        
        result = data.copy()
        
        for name, series in indicators.items():
            result[name] = series
        
        return result
    
    def _default_config(self) -> Dict:
        """Конфігурація за замовчуванням"""
        return {
            # Trend
            'ema_10': 10,
            'ema_20': 20,
            'ema_50': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'trix_period': 15,
            
            # Momentum
            'rsi_period': 14,
            'stoch_k': 14,
            'stoch_smooth_k': 3,
            'stoch_smooth_d': 3,
            'roc_period': 10,
            'momentum_period': 10,
            'williams_r_period': 14,
            'cci_period': 20,
            
            # Volatility
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'adx_period': 14,
            
            # Volume
            'volume_sma_period': 20,
            'volume_ratio_period': 20,
        }
    
    def get_performance_stats(self) -> Dict:
        """Отримання статистики продуктивності"""
        return {
            'rust_available': RUST_AVAILABLE,
            'async_enabled': self.use_async,
            'workers': self.n_workers,
            'expected_speedup': '25x' if RUST_AVAILABLE else '1x (baseline)',
            'fallback_method': 'pandas' if not RUST_AVAILABLE else 'rust+pandas',
        }


# =============================================================================
# SYNCHRONOUS HELPER FUNCTION
# =============================================================================

def calculate_all_indicators(data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Синхронна обгортка для розрахунку всіх індикаторів
    
    Args:
        data: DataFrame with OHLCV data
        config: Configuration dict
    
    Returns:
        DataFrame з усіма індикаторами
    """
    calculator = OptimizedIndicatorCalculator(use_async=True, n_workers=4)
    
    try:
        loop = asyncio.get_running_loop()
        # Event loop вже запущений - використовуємо простий розрахунок
        logger.warning("Event loop already running, using sequential calculation")
        return _sequential_calculate(data, config)
    except RuntimeError:
        # Немає event loop - створюємо новий
        return asyncio.run(calculator.calculate_all_to_dataframe(data, config))


def _sequential_calculate(data: pd.DataFrame, config: Optional[Dict]) -> pd.DataFrame:
    """Fallback для випадку коли event loop вже запущений"""
    from .trend import calculate_ema, calculate_macd, calculate_trix
    from .momentum import calculate_rsi, calculate_stochastic, calculate_roc, calculate_momentum
    from .volatility import calculate_atr, calculate_bollinger_bands, calculate_adx
    from .volume import calculate_obv, calculate_vwap
    
    if config is None:
        config = OptimizedIndicatorCalculator()._default_config()
    
    result = data.copy()
    
    # Trend
    result['EMA_10'] = calculate_ema(data, config['ema_10'])
    result['EMA_20'] = calculate_ema(data, config['ema_20'])
    result['EMA_50'] = calculate_ema(data, config['ema_50'])
    
    macd, macd_signal, macd_hist = calculate_macd(data, config['macd_fast'], config['macd_slow'], config['macd_signal'])
    result['MACD'] = macd
    result['MACD_Signal'] = macd_signal
    result['MACD_Histogram'] = macd_hist
    
    result['TRIX'] = calculate_trix(data, config['trix_period'])
    
    # Momentum
    result['RSI'] = calculate_rsi(data, config['rsi_period'])
    stoch_k, stoch_d = calculate_stochastic(data, config['stoch_k'], config['stoch_smooth_k'], config['stoch_smooth_d'])
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d
    result['ROC'] = calculate_roc(data, config['roc_period'])
    result['Momentum'] = calculate_momentum(data, config['momentum_period'])
    
    # Volatility
    result['ATR'] = calculate_atr(data, config['atr_period'])
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data, config['bb_period'], config['bb_std'])
    result['BB_Upper'] = bb_upper
    result['BB_Middle'] = bb_middle
    result['BB_Lower'] = bb_lower
    result['ADX'] = calculate_adx(data, config['adx_period'])
    
    # Volume
    result['OBV'] = calculate_obv(data)
    result['VWAP'] = calculate_vwap(data)
    
    return result
