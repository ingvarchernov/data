"""Допоміжні функції для зворотної сумісності та утиліти стратегії."""
import math
from importlib import import_module
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from . import UnifiedBinanceLoader


def _resolve_unified_loader():
    """Отримати клас UnifiedBinanceLoader без циклічних імпортів."""
    module = import_module('intelligent_sys')
    return module.UnifiedBinanceLoader


async def get_historical_data(
    symbol: str,
    interval: str,
    days_back: int,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    use_public: bool = True
) -> pd.DataFrame:
    """Функція для зворотної сумісності з data_extraction.py"""
    UnifiedBinanceLoader = _resolve_unified_loader()
    loader = UnifiedBinanceLoader(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        use_public_data=use_public
    )
    
    try:
        return await loader.get_historical_data(symbol, interval, days_back)
    finally:
        await loader.close()


def calculate_signal_confidence(predicted_change: float, df: pd.DataFrame) -> float:
    """Оцінка впевненості прогнозу на основі ринкових метрик."""
    if df is None or df.empty:
        return 0.5

    try:
        closes = df['close'].astype(float)
        returns = closes.pct_change().tail(30).dropna()
        volatility = float(returns.std()) if not returns.empty else 0.0
        if not math.isfinite(volatility) or volatility <= 1e-6:
            volatility = 1e-3

        trend_strength = abs(predicted_change) / volatility
        base_confidence = 0.5 + 0.35 * math.tanh(trend_strength)

        momentum = float(closes.pct_change(periods=5).iloc[-1]) if len(closes) > 5 else 0.0
        if math.isfinite(momentum):
            alignment = predicted_change * momentum * 100
            base_confidence += 0.1 * math.tanh(alignment)

        volume_ratio = 0.0
        if 'volume' in df.columns and len(df) > 10:
            avg_volume = float(df['volume'].rolling(10).mean().iloc[-1])
            current_volume = float(df['volume'].iloc[-1])
            if math.isfinite(avg_volume) and avg_volume > 0:
                volume_ratio = (current_volume - avg_volume) / avg_volume
        base_confidence += 0.05 * math.tanh(volume_ratio * 3)

        volatility_penalty = 0.05 * math.tanh(volatility * 50)
        base_confidence -= volatility_penalty

        return float(max(0.05, min(0.98, base_confidence)))

    except Exception:
        fallback = 0.5 + 0.3 * math.tanh(abs(predicted_change) * 50)
        return float(max(0.05, min(0.95, fallback)))


async def save_ohlcv_to_db(
    db_manager,
    symbol: str,
    interval: str,
    days_back: int = 7
) -> int:
    """Функція для зворотної сумісності з binance_loader.py"""
    UnifiedBinanceLoader = _resolve_unified_loader()
    loader = UnifiedBinanceLoader(use_public_data=True)
    
    try:
        return await loader.save_to_database(db_manager, symbol, interval, days_back)
    finally:
        await loader.close()