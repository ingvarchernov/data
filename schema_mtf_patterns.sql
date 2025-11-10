-- Multi-Timeframe Pattern Database Schema

-- Patterns table - stores all detected patterns
CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    price FLOAT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional pattern details
    strength VARCHAR(20),
    mtf_score FLOAT,
    
    -- Indexes for fast queries
    INDEX idx_symbol (symbol),
    INDEX idx_timestamp (timestamp),
    INDEX idx_timeframe (timeframe),
    INDEX idx_pattern_type (pattern_type),
    INDEX idx_created_at (created_at)
);

-- Pattern statistics - for recurring pattern tracking
CREATE TABLE IF NOT EXISTS pattern_stats (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Statistics
    occurrence_count INT DEFAULT 1,
    avg_confidence FLOAT,
    last_seen TIMESTAMP,
    first_seen TIMESTAMP,
    
    -- Success tracking (for future backtesting)
    success_count INT DEFAULT 0,
    total_trades INT DEFAULT 0,
    avg_profit FLOAT DEFAULT 0,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(symbol, pattern_type, direction, timeframe),
    
    -- Indexes
    INDEX idx_occurrence (occurrence_count),
    INDEX idx_last_seen (last_seen)
);

-- MTF Confluence signals
CREATE TABLE IF NOT EXISTS mtf_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    mtf_score FLOAT NOT NULL,
    confluence_pct FLOAT NOT NULL,
    dominant_direction VARCHAR(10) NOT NULL,
    timeframes_count INT NOT NULL,
    total_patterns INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Signal metadata
    sent_to_telegram BOOLEAN DEFAULT FALSE,
    telegram_sent_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_mtf_symbol (symbol),
    INDEX idx_mtf_score (mtf_score),
    INDEX idx_mtf_timestamp (timestamp)
);

-- Pattern chart data (for visualization)
CREATE TABLE IF NOT EXISTS pattern_chart_snapshots (
    id SERIAL PRIMARY KEY,
    pattern_id INT REFERENCES patterns(id) ON DELETE CASCADE,
    
    -- OHLCV data around pattern (JSON)
    candles_data JSONB,
    
    -- Indicators data (JSON)
    indicators_data JSONB,
    
    -- Pattern specific data
    pattern_coordinates JSONB,  -- {start_idx, end_idx, key_points}
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Views for quick queries

-- Recent patterns view
CREATE OR REPLACE VIEW recent_patterns AS
SELECT 
    p.symbol,
    p.pattern_type,
    p.direction,
    p.timeframe,
    p.confidence,
    p.price,
    p.timestamp,
    ps.occurrence_count
FROM patterns p
LEFT JOIN pattern_stats ps 
    ON p.symbol = ps.symbol 
    AND p.pattern_type = ps.pattern_type 
    AND p.direction = ps.direction
    AND p.timeframe = ps.timeframe
WHERE p.timestamp >= NOW() - INTERVAL '7 days'
ORDER BY p.timestamp DESC;

-- Top recurring patterns view
CREATE OR REPLACE VIEW top_recurring_patterns AS
SELECT 
    symbol,
    pattern_type,
    direction,
    timeframe,
    occurrence_count,
    avg_confidence,
    last_seen,
    CASE 
        WHEN total_trades > 0 THEN (success_count::FLOAT / total_trades * 100)
        ELSE 0
    END as success_rate
FROM pattern_stats
WHERE occurrence_count >= 5
ORDER BY occurrence_count DESC, avg_confidence DESC;

-- MTF confluence signals view
CREATE OR REPLACE VIEW top_mtf_signals AS
SELECT 
    symbol,
    mtf_score,
    confluence_pct,
    dominant_direction,
    timeframes_count,
    total_patterns,
    timestamp,
    sent_to_telegram
FROM mtf_signals
WHERE timestamp >= NOW() - INTERVAL '24 hours'
    AND confluence_pct >= 60
    AND timeframes_count >= 3
ORDER BY mtf_score DESC, confluence_pct DESC;
