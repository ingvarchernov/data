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
    directional_accuracy DOUBLE PRECISION,
    real_mae DOUBLE PRECISION,
    real_mape DOUBLE PRECISION,
    PRIMARY KEY (symbol_id, interval_id, fold, epoch)
);

-- Додавання нової колонки до існуючої таблиці (якщо потрібно)
-- ALTER TABLE training_history ADD COLUMN directional_accuracy DOUBLE PRECISION;

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

-- Таблиці для системи моніторингу
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    cpu_percent DOUBLE PRECISION,
    memory_percent DOUBLE PRECISION,
    disk_usage DOUBLE PRECISION,
    gpu_memory_used DOUBLE PRECISION,
    gpu_memory_total DOUBLE PRECISION,
    gpu_utilization DOUBLE PRECISION
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    interval VARCHAR(16) NOT NULL,
    model_type VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    training_time DOUBLE PRECISION,
    epochs INTEGER,
    final_loss DOUBLE PRECISION,
    final_val_loss DOUBLE PRECISION,
    final_mae DOUBLE PRECISION,
    final_val_mae DOUBLE PRECISION,
    directional_accuracy DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    val_mape DOUBLE PRECISION
);

CREATE TABLE prediction_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    interval VARCHAR(16) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    predicted_price DOUBLE PRECISION,
    actual_price DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    directional_correct BOOLEAN,
    confidence_score DOUBLE PRECISION
);

-- Індекси для таблиць моніторингу
CREATE INDEX idx_system_metrics_timestamp ON system_metrics (timestamp);
CREATE INDEX idx_model_metrics_symbol_timestamp ON model_metrics (symbol, timestamp);
CREATE INDEX idx_prediction_metrics_symbol_timestamp ON prediction_metrics (symbol, timestamp);

-- Таблиця для фундаментальних даних
CREATE TABLE fundamental_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    news_sentiment_score FLOAT,
    news_sentiment_confidence FLOAT,
    news_volume INTEGER,
    social_sentiment_score FLOAT,
    social_sentiment_confidence FLOAT,
    social_volume INTEGER,
    active_addresses INTEGER,
    transaction_count INTEGER,
    transaction_volume FLOAT,
    large_transactions INTEGER,
    whale_activity FLOAT,
    network_hashrate FLOAT,
    gas_price FLOAT,
    aggregate_sentiment FLOAT,
    sentiment_momentum FLOAT,
    fear_greed_index FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timestamp, symbol)
);

-- Індекси для фундаментальних даних
CREATE INDEX idx_fundamental_timestamp_symbol ON fundamental_data (timestamp, symbol);
CREATE INDEX idx_fundamental_symbol_timestamp ON fundamental_data (symbol, timestamp DESC);

-- Таблиця для торгових сигналів
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    confidence FLOAT NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    strategy VARCHAR(50),
    prediction_source VARCHAR(50) DEFAULT 'technical', -- ml_model, technical
    status VARCHAR(20) DEFAULT 'generated', -- generated, executed, rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    notes TEXT
);

-- Таблиця для відкритих позицій
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG, SHORT
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50),
    status VARCHAR(20) DEFAULT 'open', -- open, closed, stopped
    signal_id INTEGER REFERENCES trading_signals(id),
    metadata JSONB
);

-- Таблиця для закритих угод
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG, SHORT
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pnl DECIMAL(20, 8) NOT NULL, -- Прибуток/збиток в доларах
    pnl_percentage DECIMAL(10, 4), -- Прибуток/збиток у відсотках
    strategy VARCHAR(50),
    exit_reason VARCHAR(50), -- take_profit, stop_loss, manual, signal
    position_id INTEGER REFERENCES positions(id),
    signal_id INTEGER REFERENCES trading_signals(id),
    fees DECIMAL(20, 8) DEFAULT 0,
    metadata JSONB
);

-- Індекси для торгів
CREATE INDEX idx_trading_signals_symbol_created ON trading_signals (symbol, created_at DESC);
CREATE INDEX idx_trading_signals_status ON trading_signals (status);
CREATE INDEX idx_positions_symbol_status ON positions (symbol, status);
CREATE INDEX idx_trades_symbol_exit_time ON trades (symbol, exit_time DESC);
CREATE INDEX idx_trades_pnl ON trades (pnl DESC);
CREATE INDEX idx_trades_strategy ON trades (strategy);