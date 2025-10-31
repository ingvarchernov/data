-- ============================================================================
-- 📊 TRADING DATABASE SCHEMA
-- ============================================================================
-- Повна схема бази даних для торгової системи
-- Включає: позиції, трейди, історію змін, аналітику PnL
-- ============================================================================

-- Видалення існуючих таблиць (якщо потрібно почати з чистого листа)
DROP TABLE IF EXISTS position_history CASCADE;
DROP TABLE IF EXISTS trades CASCADE;
DROP TABLE IF EXISTS positions CASCADE;
DROP TABLE IF EXISTS trading_sessions CASCADE;

-- ============================================================================
-- 1. TRADING SESSIONS - торгові сесії
-- ============================================================================
CREATE TABLE trading_sessions (
    id SERIAL PRIMARY KEY,
    session_start TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP WITH TIME ZONE,
    initial_balance DECIMAL(20, 8) NOT NULL,
    final_balance DECIMAL(20, 8),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    max_drawdown DECIMAL(10, 4) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'stopped')),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_status ON trading_sessions(status);
CREATE INDEX idx_sessions_start ON trading_sessions(session_start DESC);

-- ============================================================================
-- 2. POSITIONS - позиції (відкриті/закриті)
-- ============================================================================
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES trading_sessions(id) ON DELETE SET NULL,
    
    -- Основна інформація
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'liquidated')),
    
    -- Параметри входу
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    leverage INTEGER NOT NULL DEFAULT 1,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Параметри виходу
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMP WITH TIME ZONE,
    exit_reason VARCHAR(100),  -- Без constraint - дозволяємо будь-які причини
    
    -- Захист (SL/TP)
    stop_loss_price DECIMAL(20, 8),
    take_profit_price DECIMAL(20, 8),
    trailing_stop_active BOOLEAN DEFAULT FALSE,
    trailing_stop_distance DECIMAL(10, 4),
    best_price DECIMAL(20, 8),  -- Найкраща ціна для trailing stop
    
    -- PnL
    realized_pnl DECIMAL(20, 8),
    realized_pnl_pct DECIMAL(10, 4),
    fees DECIMAL(20, 8) DEFAULT 0,
    
    -- ML прогноз (що спричинив відкриття)
    ml_prediction VARCHAR(10) CHECK (ml_prediction IN ('UP', 'DOWN')),
    ml_confidence DECIMAL(5, 4),
    ml_features JSONB,  -- Збереження фічей для аналізу
    
    -- Binance дані
    binance_order_id BIGINT,
    binance_position_id BIGINT,
    
    -- Метадані
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Індекси для швидкого пошуку
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_side ON positions(side);
CREATE INDEX idx_positions_entry_time ON positions(entry_time DESC);
CREATE INDEX idx_positions_exit_time ON positions(exit_time DESC);
CREATE INDEX idx_positions_session ON positions(session_id);
CREATE INDEX idx_positions_open ON positions(symbol, status) WHERE status = 'open';

-- ============================================================================
-- 3. TRADES - окремі трейди (заповнення ордерів)
-- ============================================================================
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id) ON DELETE CASCADE,
    
    -- Основна інформація
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    trade_type VARCHAR(20) NOT NULL CHECK (trade_type IN (
        'OPEN', 'CLOSE', 'PARTIAL_CLOSE', 'ADD', 'REDUCE'
    )),
    
    -- Параметри трейду
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    quote_quantity DECIMAL(20, 8) NOT NULL,  -- Вартість в USDT
    
    -- PnL для закриваючих трейдів
    realized_pnl DECIMAL(20, 8),
    fee DECIMAL(20, 8) DEFAULT 0,
    fee_asset VARCHAR(10) DEFAULT 'USDT',
    
    -- Binance дані
    binance_trade_id BIGINT UNIQUE,
    binance_order_id BIGINT,
    is_maker BOOLEAN DEFAULT FALSE,
    
    -- Час
    trade_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Індекси
CREATE INDEX idx_trades_position ON trades(position_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_time ON trades(trade_time DESC);
CREATE INDEX idx_trades_binance_id ON trades(binance_trade_id);

-- ============================================================================
-- 4. POSITION_HISTORY - історія змін позицій
-- ============================================================================
CREATE TABLE position_history (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id) ON DELETE CASCADE,
    
    -- Що змінилося
    event_type VARCHAR(30) NOT NULL CHECK (event_type IN (
        'OPEN', 'CLOSE', 'UPDATE_SL', 'UPDATE_TP', 
        'ACTIVATE_TRAILING', 'UPDATE_TRAILING', 
        'PNL_UPDATE', 'REVERSE', 'FORCE_CLOSE'
    )),
    
    -- Дані до зміни
    old_status VARCHAR(20),
    old_stop_loss DECIMAL(20, 8),
    old_take_profit DECIMAL(20, 8),
    old_trailing_active BOOLEAN,
    
    -- Дані після зміни
    new_status VARCHAR(20),
    new_stop_loss DECIMAL(20, 8),
    new_take_profit DECIMAL(20, 8),
    new_trailing_active BOOLEAN,
    
    -- Поточні дані
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 4),
    
    -- Причина зміни
    reason TEXT,
    triggered_by VARCHAR(20) CHECK (triggered_by IN (
        'SYSTEM', 'USER', 'ML_SIGNAL', 'WEBSOCKET', 'MONITOR'
    )),
    
    -- Час
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Індекси
CREATE INDEX idx_history_position ON position_history(position_id);
CREATE INDEX idx_history_event ON position_history(event_type);
CREATE INDEX idx_history_time ON position_history(event_time DESC);

-- ============================================================================
-- 5. PNL_ANALYTICS - агреговані дані для аналітики
-- ============================================================================
CREATE TABLE pnl_analytics (
    id SERIAL PRIMARY KEY,
    
    -- Період
    period_type VARCHAR(10) NOT NULL CHECK (period_type IN ('DAILY', 'WEEKLY', 'MONTHLY')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Загальна статистика
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    
    -- PnL
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    gross_profit DECIMAL(20, 8) DEFAULT 0,
    gross_loss DECIMAL(20, 8) DEFAULT 0,
    profit_factor DECIMAL(10, 2),
    
    -- Середні значення
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    avg_trade_duration INTEGER,  -- хвилини
    
    -- Екстремуми
    best_trade DECIMAL(20, 8),
    worst_trade DECIMAL(20, 8),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    max_drawdown DECIMAL(10, 4),
    
    -- По символах (топ 5)
    top_symbols JSONB,
    
    -- Час
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(period_type, period_start)
);

CREATE INDEX idx_analytics_period ON pnl_analytics(period_type, period_start DESC);

-- ============================================================================
-- ТРИГЕРИ ДЛЯ АВТОМАТИЧНОГО ОНОВЛЕННЯ updated_at
-- ============================================================================

-- Функція для оновлення updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Тригери
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON trading_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ФУНКЦІЇ ДЛЯ АНАЛІТИКИ
-- ============================================================================

-- Функція для розрахунку PnL по символу
CREATE OR REPLACE FUNCTION calculate_symbol_pnl(
    p_symbol VARCHAR,
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE (
    symbol VARCHAR,
    total_trades BIGINT,
    winning_trades BIGINT,
    total_pnl NUMERIC,
    win_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_symbol,
        COUNT(*)::BIGINT,
        COUNT(*) FILTER (WHERE realized_pnl > 0)::BIGINT,
        COALESCE(SUM(realized_pnl), 0),
        CASE 
            WHEN COUNT(*) > 0 THEN 
                (COUNT(*) FILTER (WHERE realized_pnl > 0)::NUMERIC / COUNT(*)::NUMERIC * 100)
            ELSE 0 
        END
    FROM positions
    WHERE 
        positions.symbol = p_symbol
        AND status = 'closed'
        AND (p_start_date IS NULL OR exit_time >= p_start_date)
        AND (p_end_date IS NULL OR exit_time <= p_end_date);
END;
$$ LANGUAGE plpgsql;

-- Функція для отримання відкритих позицій
CREATE OR REPLACE FUNCTION get_open_positions()
RETURNS TABLE (
    position_id INTEGER,
    symbol VARCHAR,
    side VARCHAR,
    entry_price NUMERIC,
    quantity NUMERIC,
    leverage INTEGER,
    entry_time TIMESTAMP WITH TIME ZONE,
    stop_loss_price NUMERIC,
    take_profit_price NUMERIC,
    trailing_stop_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id,
        p.symbol,
        p.side,
        p.entry_price,
        p.quantity,
        p.leverage,
        p.entry_time,
        p.stop_loss_price,
        p.take_profit_price,
        p.trailing_stop_active
    FROM positions p
    WHERE p.status = 'open'
    ORDER BY p.entry_time DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ПОЧАТКОВІ ДАНІ
-- ============================================================================

-- Створюємо початкову сесію
INSERT INTO trading_sessions (initial_balance, status, notes)
VALUES (10000.00, 'active', 'Початкова торгова сесія');

-- ============================================================================
-- ПРАВА ДОСТУПУ (опціонально, якщо є окремий користувач)
-- ============================================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_bot_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_bot_user;

-- ============================================================================
-- КОМЕНТАРІ
-- ============================================================================
COMMENT ON TABLE positions IS 'Торгові позиції (відкриті/закриті)';
COMMENT ON TABLE trades IS 'Окремі трейди (заповнення ордерів)';
COMMENT ON TABLE position_history IS 'Історія всіх змін позицій';
COMMENT ON TABLE trading_sessions IS 'Торгові сесії';
COMMENT ON TABLE pnl_analytics IS 'Агрегована аналітика PnL';

COMMENT ON COLUMN positions.ml_features IS 'JSON з ML фічами що використовувалися для прогнозу';
COMMENT ON COLUMN positions.trailing_stop_active IS 'Чи активний trailing stop для цієї позиції';
COMMENT ON COLUMN positions.best_price IS 'Найкраща ціна досягнута позицією (для trailing stop)';

-- ============================================================================
-- ВИВЕДЕННЯ СТАТИСТИКИ
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Схема бази даних створена успішно!';
    RAISE NOTICE '';
    RAISE NOTICE '📊 Створені таблиці:';
    RAISE NOTICE '   • trading_sessions - торгові сесії';
    RAISE NOTICE '   • positions - позиції';
    RAISE NOTICE '   • trades - трейди';
    RAISE NOTICE '   • position_history - історія змін';
    RAISE NOTICE '   • pnl_analytics - аналітика';
    RAISE NOTICE '';
    RAISE NOTICE '🔧 Створені функції:';
    RAISE NOTICE '   • calculate_symbol_pnl() - розрахунок PnL по символу';
    RAISE NOTICE '   • get_open_positions() - отримання відкритих позицій';
    RAISE NOTICE '';
    RAISE NOTICE '📈 Готово до використання!';
END $$;
