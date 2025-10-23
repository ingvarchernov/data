-- Таблиці для системи торгівлі
-- Створено для інтеграції сигналів з Telegram та аналізу торгівлі

-- Таблиця торгових сигналів
CREATE TABLE IF NOT EXISTS trading_signals (
    signal_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL, -- BUY, SELL, CLOSE
    confidence DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    entry_price DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    strategy VARCHAR(50) DEFAULT 'unknown',
    prediction_source VARCHAR(50) DEFAULT 'technical',
    status VARCHAR(20) DEFAULT 'generated', -- generated, executed, expired
    notes TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Індекси для trading_signals
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_status ON trading_signals(status);
CREATE INDEX IF NOT EXISTS idx_trading_signals_created_at ON trading_signals(created_at DESC);

-- Таблиця відкритих позицій
CREATE TABLE IF NOT EXISTS positions (
    position_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG, SHORT
    entry_price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    strategy VARCHAR(50) DEFAULT 'unknown',
    status VARCHAR(20) DEFAULT 'open', -- open, closed, cancelled
    signal_id INTEGER REFERENCES trading_signals(signal_id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Індекси для positions
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_signal_id ON positions(signal_id);

-- Таблиця закритих угод
CREATE TABLE IF NOT EXISTS trades (
    trade_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG, SHORT
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE NOT NULL,
    pnl DECIMAL(20,8) NOT NULL, -- Прибуток/збиток в базовій валюті
    pnl_percentage DECIMAL(10,4) NOT NULL, -- Прибуток/збиток у відсотках
    strategy VARCHAR(50) DEFAULT 'unknown',
    exit_reason VARCHAR(50) DEFAULT 'manual', -- manual, stop_loss, take_profit, signal
    position_id INTEGER REFERENCES positions(position_id),
    signal_id INTEGER REFERENCES trading_signals(signal_id),
    fees DECIMAL(20,8) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Індекси для trades
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id);

-- Тригери для автоматичного оновлення updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trading_signals_updated_at
    BEFORE UPDATE ON trading_signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();