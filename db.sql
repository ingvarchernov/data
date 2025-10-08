-- Створення таблиці symbols
CREATE TABLE symbols (
    symbol_id SERIAL PRIMARY KEY,
    symbol VARCHAR(32) UNIQUE NOT NULL
);

-- Створення таблиці intervals
CREATE TABLE intervals (
    interval_id SERIAL PRIMARY KEY,
    interval VARCHAR(16) UNIQUE NOT NULL
);

-- Створення таблиці historical_data
CREATE TABLE historical_data (
    data_id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id),
    interval_id INTEGER NOT NULL REFERENCES intervals(interval_id),
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    quote_av DOUBLE PRECISION,
    trades INTEGER,
    tb_base_av DOUBLE PRECISION,
    tb_quote_av DOUBLE PRECISION,
    UNIQUE(symbol_id, interval_id, timestamp)
);

-- Створення таблиці technical_indicators
CREATE TABLE technical_indicators (
    data_id INTEGER PRIMARY KEY REFERENCES historical_data(data_id),
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    upper_band DOUBLE PRECISION,
    lower_band DOUBLE PRECISION,
    stoch DOUBLE PRECISION,
    stoch_signal DOUBLE PRECISION,
    ema DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    cci DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    volume_pct DOUBLE PRECISION,
    close_lag1 DOUBLE PRECISION,
    close_lag2 DOUBLE PRECISION,
    close_diff DOUBLE PRECISION,
    log_return DOUBLE PRECISION,
    hour_norm DOUBLE PRECISION,
    day_norm DOUBLE PRECISION,
    adx DOUBLE PRECISION,
    vwap DOUBLE PRECISION
);

-- Створення таблиці normalized_data
CREATE TABLE normalized_data (
    data_id INTEGER NOT NULL REFERENCES historical_data(data_id),
    feature VARCHAR(64) NOT NULL,
    normalized_value DOUBLE PRECISION,
    PRIMARY KEY (data_id, feature)
);

-- Створення таблиці training_history
CREATE TABLE training_history (
    symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id),
    interval_id INTEGER NOT NULL REFERENCES intervals(interval_id),
    fold INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    loss DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    val_loss DOUBLE PRECISION,
    val_mae DOUBLE PRECISION,
    val_mape DOUBLE PRECISION,
    real_mae DOUBLE PRECISION,
    real_mape DOUBLE PRECISION,
    PRIMARY KEY (symbol_id, interval_id, fold, epoch)
);

-- Створення таблиці predictions
CREATE TABLE predictions (
    symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id),
    interval_id INTEGER NOT NULL REFERENCES intervals(interval_id),
    timestamp TIMESTAMP NOT NULL,
    last_price DOUBLE PRECISION,
    predicted_price DOUBLE PRECISION,
    fold_1_prediction DOUBLE PRECISION,
    fold_2_prediction DOUBLE PRECISION,
    fold_3_prediction DOUBLE PRECISION,
    fold_4_prediction DOUBLE PRECISION,
    fold_5_prediction DOUBLE PRECISION,
    PRIMARY KEY (symbol_id, interval_id, timestamp)
);

-- Створення таблиці scaler_stats
CREATE TABLE scaler_stats (
    symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id),
    interval_id INTEGER NOT NULL REFERENCES intervals(interval_id),
    target_mean DOUBLE PRECISION,
    target_std DOUBLE PRECISION,
    PRIMARY KEY (symbol_id, interval_id)
);

-- Додаткові індекси для швидкості
CREATE INDEX idx_historical_data_timestamp ON historical_data (timestamp);
CREATE INDEX idx_predictions_symbol_interval_timestamp ON predictions (symbol_id, interval_id, timestamp);