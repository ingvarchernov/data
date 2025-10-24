SELECTED_FEATURES = [
    # Base OHLCV
    'open',
    'volume',
    'returns',
    'log_returns',
    
    # Trend indicators (SMA/EMA) - without sma_200 (too large period)
    'sma_10', 'sma_20', 'sma_50',
    'sma_10_ratio', 'sma_20_ratio', 'sma_50_ratio',
    'ema_12', 'ema_20', 'ema_26', 'ema_50',
    'ema_12_ratio', 'ema_20_ratio', 'ema_26_ratio', 'ema_50_ratio',
    
    # Momentum
    'rsi_7', 'rsi_14', 'rsi_28',
    'roc_5', 'roc_10', 'roc_20',
    'price_momentum',
    'acceleration',
    
    # Volatility
    'atr_7', 'atr_14', 'atr_21',
    'close_std_5', 'close_std_10', 'close_std_20', 'close_std_30',
    'hvol_10', 'hvol_20', 'hvol_30',
    
    # Bollinger Bands
    'bb_width_20', 'bb_width_50',
    'bb_percent_20', 'bb_percent_50',
    
    # Volume
    'obv',
    'vwap',
    'volume_mean_5', 'volume_mean_10', 'volume_mean_20', 'volume_mean_30',
    'volume_ratio',
    'volume_momentum',
    
    # Distance from means
    'dist_from_mean_5', 'dist_from_mean_10', 'dist_from_mean_20',
    
    # Candle patterns
    'body',
    'upper_wick',
    'lower_wick',
    'body_ratio',
]
