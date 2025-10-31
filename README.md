# 🤖 Crypto Trading System - Stable v1.0

> **Automated ML-based crypto futures trading bot** with risk management, multi-timeframe analysis, and real-time position monitoring.

[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PostgreSQL](https://img.shields.io/badge/postgresql-14+-blue)]()
[![Binance](https://img.shields.io/badge/binance-futures-yellow)]()

---

## 🎯 Key Features

- ✅ **Multi-Timeframe Analysis**: 15m + 1h consensus-based signals (MTF)
- 🤖 **ML Predictions**: Random Forest with 82 technical indicators (Rust-accelerated)
- 📊 **Strategy Selector**: Automatic market regime detection (trend/range/consolidation)
- 💾 **PostgreSQL Database**: Position tracking, trade history, performance analytics
- 🛡️ **Risk Management**: SL/TP, trailing stop, force close protection, position sizing
- 📡 **Real-time Monitoring**: WebSocket integration, 15s position checks
- 🔄 **Online Learning**: Incremental model retraining after failed trades
- 📈 **Volatility Filter**: Dynamic pair selection based on market activity

---

## 📈 Performance (Testnet)

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|---------|
| **Win Rate** | 2.8% | 50.0% | +47.2% ✅ |
| **Force Close** | 94.4% | 50.0% | -44.4% ✅ |
| **Avg PnL** | -$3.50 | +$5.33 | +$8.83 ✅ |
| **Profit Factor** | - | 5.48 | Excellent ✅ |

**Last Session Stats**:
- Avg WIN: $13.04 (+25.17%)
- Avg LOSS: $2.38 (-4.76%)
- Trailing Stop Working: ✅ Captured +25% instead of +2.5%

---

## 🏗️ Architecture

```
data/
├── core/                      # Trading engine core modules
│   ├── trading_bot.py         # Main bot orchestrator (signals → positions)
│   ├── position_manager.py    # Order execution, SL/TP management
│   ├── position_monitor.py    # Real-time P&L monitoring, force close
│   ├── ml_predictor.py        # ML model inference (Random Forest)
│   ├── analytics.py           # Performance metrics, session stats
│   ├── volatility_scanner.py  # Market activity scoring (0-100)
│   └── trades_synchronizer.py # Binance ↔ DB sync on startup
│
├── strategies/                # Trading strategies (local only)
│   ├── strategy_selector.py  # Market regime detection + strategy routing
│   ├── mean_reversion.py     # Range-bound markets strategy
│   ├── trend_following.py    # Trending markets strategy
│   └── base.py               # Base strategy interface
│
├── training/                  # ML training scripts (local only)
│   ├── simple_trend_classifier.py  # Random Forest trainer
│   ├── online_learning.py          # Incremental retraining system
│   ├── rust_features.py            # Rust indicators wrapper
│   └── batch_train_rf.py           # Batch training for all pairs
│
├── optimized/                 # Database layer
│   ├── database/
│   │   ├── connection.py      # PostgreSQL connection pooling
│   │   ├── positions.py       # Positions CRUD operations
│   │   └── cache.py           # Redis/in-memory cache
│   └── indicators/            # Technical analysis modules
│
├── fast_indicators/           # Rust-accelerated indicators
│   ├── src/lib.rs            # RSI, MACD, Bollinger, ATR, etc.
│   └── Cargo.toml
│
├── utils/                     # Utility scripts
│   ├── check_db.py           # Database diagnostics
│   ├── check_orders.py       # Active orders inspector
│   ├── close_all_positions.py # Emergency position closer
│   └── analyze_failures.py   # Trade failure analysis
│
├── main.py                    # Bot entry point
├── config.py                  # Configuration (API keys, risk params)
├── preflight_check.py         # Pre-launch validation
├── incremental_retrain.py     # Model retraining scheduler
├── telegram_bot.py            # Telegram notifications
├── websocket_manager.py       # Binance WebSocket handler
├── mtf_analyzer.py            # Multi-timeframe signal aggregator
└── database_schema.sql        # PostgreSQL schema definition
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# System requirements
- Python 3.10+
- PostgreSQL 14+
- Rust 1.70+ (for fast_indicators)
- Git

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv postgresql postgresql-contrib rustc cargo
```

### 2. Clone & Setup

```bash
# Clone repository
git clone https://github.com/ingvarchernov/data.git
cd data

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Build Rust indicators
cd fast_indicators
cargo build --release
cd ..
```

### 3. Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required `.env` variables:
```bash
# Binance API (testnet or production)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here
BINANCE_TESTNET=true  # Set false for production

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. Initialize Database

```bash
# Create PostgreSQL database
sudo -u postgres psql
CREATE DATABASE trading_bot;
CREATE USER trading_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q

# Run schema migration
python init_database.py
```

### 5. Train Models

```bash
# Train models for all pairs (first time)
python -c "from training.batch_train_rf import train_all_models; train_all_models()"

# Models will be saved to models/simple_trend_<SYMBOL>/
```

### 6. Run Bot

```bash
# Pre-flight check (validates setup)
python preflight_check.py

# Start trading bot
python main.py

# Or run in background
nohup python main.py > bot.log 2>&1 &
```

---

## ⚙️ Configuration

Edit `config.py` to adjust trading parameters:

```python
# Trading pairs (5 best performers)
TRADING_SYMBOLS = [
    'BTCUSDT',   # 93% accuracy (15m), 86% (1h)
    'TRXUSDT',   # 86% accuracy (15m), 79% (1h)
    'SOLUSDT',   # 76% accuracy (15m), 67% (1h)
    'XRPUSDT',   # 70% accuracy (15m), 65% (1h)
    'ETHUSDT',   # 67% accuracy (15m), 65% (1h)
]

# Risk Management
TRADING_CONFIG = {
    'stop_loss_pct': 0.020,      # 2.0% (50% loss on 25x leverage)
    'take_profit_pct': 0.025,    # 2.5% (62.5% profit on 25x leverage)
    'leverage': 25,
    'position_size_usd': 50,     # Base position size
    'min_confidence': 0.70,      # 70% minimum ML confidence
    'min_consensus': 0.85,       # 85% MTF agreement required
}

# Position Monitor
POSITION_MONITOR = {
    'check_interval': 15,        # Check every 15 seconds
    'max_loss_pct': 0.05,        # 5% max loss (125% on leverage)
    'force_close_threshold': 0.04, # 4% emergency close (100% loss)
}

# Trailing Stop
TRAILING_STOP = {
    'activation_profit': 0.015,  # Activate at +1.5% profit
    'trail_distance': 0.50,      # Trail 50% from peak
}

# Multi-Timeframe
MTF_CONFIG = {
    '15m': {'weight': 0.40, 'interval': '15m', 'periods': 96},
    '1h':  {'weight': 0.60, 'interval': '1h', 'periods': 24},
}
```

---

## 🔍 How It Works

### 1. Signal Generation Flow

```
┌─────────────┐
│ Market Data │ (Binance WebSocket + REST API)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Volatility Scanner  │ (Filter low-activity pairs)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ ML Predictor        │ (82 Rust indicators → Random Forest)
│ • 15m timeframe     │
│ • 1h timeframe      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ MTF Analyzer        │ (Weighted consensus: 15m=40%, 1h=60%)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Strategy Selector   │ (Detect regime → route to strategy)
│ • Trend Following   │ (strong trend markets)
│ • Mean Reversion    │ (range-bound markets)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Confidence Filter   │ (min 70% confidence, 85% consensus)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Position Manager    │ (Execute order with SL/TP)
└─────────────────────┘
```

### 2. Position Lifecycle

```
OPEN → MONITORING → CLOSE

1. OPEN:
   - Check available slots (max 9 positions)
   - Reserve 1 slot for reversals
   - Execute market order with leverage
   - Set SL/TP orders (closePosition=True)
   - Save to database

2. MONITORING (every 15s):
   - Calculate real-time P&L
   - Check force close threshold (4%)
   - Update trailing stop if activated (+1.5%)
   - Check for stale positions (24h)
   - Emergency close on critical loss

3. CLOSE:
   - TP hit: +2.5% profit
   - SL hit: -2.0% loss
   - Trailing stop: follow profit peak
   - Force close: -4% emergency
   - Manual close: user request
```

### 3. Online Learning Cycle

```
Failed Trade → Analyze → Retrain → Deploy

1. Trade closes with loss
2. Position Monitor detects failure pattern
3. Incremental retraining triggered (background)
4. New samples added: [features, actual_outcome]
5. Model updated with recent market behavior
6. Improved predictions for next trades
```

---

## 📊 Database Schema

**Positions Table** (core trading data):
```sql
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    side VARCHAR(10),                -- LONG/SHORT
    status VARCHAR(20),              -- open/closed/cancelled
    entry_price NUMERIC(20, 8),
    exit_price NUMERIC(20, 8),
    quantity NUMERIC(20, 8),
    leverage INTEGER,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    exit_reason VARCHAR(50),         -- TP/SL/FORCE_CLOSE/Trailing Stop
    realized_pnl NUMERIC(20, 8),
    realized_pnl_pct NUMERIC(10, 4),
    ml_prediction VARCHAR(10),       -- UP/DOWN
    ml_confidence NUMERIC(5, 4),     -- 0.0-1.0
    stop_loss_price NUMERIC(20, 8),
    take_profit_price NUMERIC(20, 8),
    trailing_stop_active BOOLEAN,
    best_price NUMERIC(20, 8),       -- Peak price for trailing
    binance_order_id BIGINT
);
```

---

## 🛡️ Risk Management Features

### 1. Stop Loss / Take Profit
- **SL**: 2.0% price move = 50% loss on capital (25x leverage)
- **TP**: 2.5% price move = 62.5% profit on capital
- Orders use `closePosition=True` to prevent phantom positions
- `timeInForce='GTC'` prevents order expiration

### 2. Force Close Protection
- Monitors P&L every 15 seconds
- Emergency close at 4% loss (100% capital lost)
- Prevents catastrophic drawdowns
- Blacklists failing pairs for 180 minutes

### 3. Trailing Stop
- Activates at +1.5% profit
- Trails 50% from peak price
- Locks in profits automatically
- Example: +25% profit captured instead of fixed +2.5% TP

### 4. Position Sizing
- Fixed $50 base size per position
- 25x leverage = $1250 exposure
- Max 9 concurrent positions
- 1 slot reserved for reversals

### 5. Blacklist Mechanism
- Failed pairs blocked for 3 hours
- Prevents repeated losses on bad signals
- Automatic cooldown expiration
- Force close triggers instant blacklist

---

## 📈 Performance Optimization

### Achieved Improvements

**Problem 1**: Win rate 2.8% (34 force closes out of 36 trades)
- **Root Cause**: SL too tight (1.0%), positions closed on minor noise
- **Solution**: Increased SL to 2.0%, force_close to 4.0%
- **Result**: Win rate → 50%, force close → 50% ✅

**Problem 2**: Direction mapping bug (LONG signal → SHORT position)
- **Root Cause**: Strategy returned 'LONG'/'SHORT', bot expected 'UP'/'DOWN'
- **Solution**: Added direction mapping in trading_bot.py
- **Result**: Positions now open in correct direction ✅

**Problem 3**: Phantom positions from expired SL/TP orders
- **Root Cause**: `closePosition=False` + expired orders created reverse positions
- **Solution**: Changed to `closePosition=True`, added `timeInForce='GTC'`
- **Result**: No more phantom positions ✅

**Problem 4**: Profit missed (positions closed at +2.5% when price went to +40%)
- **Root Cause**: Fixed TP without trailing
- **Solution**: Implemented trailing stop (activates at +1.5%, trails 50%)
- **Result**: Captured +25% profit on BTCUSDT ✅

---

## 🧪 Testing & Validation

### Preflight Check
```bash
python preflight_check.py
```
Validates:
- ✅ Binance API connection
- ✅ PostgreSQL database access
- ✅ ML models loaded (5 pairs × 2 timeframes)
- ✅ WebSocket connection
- ✅ Open positions count
- ✅ Account balance

### Manual Testing
```bash
# Check database positions
python utils/check_db.py

# Inspect active orders
python utils/check_orders.py

# Emergency close all
python utils/close_all_positions.py

# Analyze trade failures
python utils/analyze_failures.py
```

---

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'fast_indicators'"**
```bash
cd fast_indicators
cargo build --release
cd ..
```

**2. "PostgreSQL connection failed"**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify credentials in .env
# Test connection manually
psql -U trading_user -d trading_bot -h localhost
```

**3. "Binance API Error: Invalid API-key"**
- Check `.env` has correct `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- For testnet: ensure `BINANCE_TESTNET=true`
- Verify API key permissions (Futures trading enabled)

**4. "Models not found"**
```bash
# Train models first
python -c "from training.batch_train_rf import train_all_models; train_all_models()"
```

**5. "Force close immediately after open"**
- Check `config.py` → `force_close_threshold` (should be 0.04 = 4%)
- Verify SL is not too tight (should be 0.02 = 2.0%)
- Testnet can have extreme volatility/slippage

---

## 📚 Resources

- **Binance Futures API**: https://binance-docs.github.io/apidocs/futures/en/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Rust TA-Lib**: https://docs.rs/ta/latest/ta/
- **scikit-learn**: https://scikit-learn.org/stable/

---

## 🚧 Roadmap

- [ ] **Synchronizer Fix**: Complete Binance ↔ DB sync on startup
- [ ] **LSTM Models**: Replace Random Forest with LSTM/Transformer
- [ ] **Backtesting Engine**: Historical performance validation
- [ ] **Multi-Exchange**: Add support for Bybit, OKX
- [ ] **Adaptive SL/TP**: Dynamic based on ATR/volatility
- [ ] **Portfolio Rebalancing**: Correlation-aware position sizing
- [ ] **News Integration**: Filter trades around high-impact events

---

## ⚠️ Disclaimer

**This software is for educational purposes only.**

- Cryptocurrency trading involves substantial risk of loss.
- Past performance does not guarantee future results.
- Test thoroughly on testnet before using real funds.
- Use at your own risk. The authors assume no liability for financial losses.
- Always start with small position sizes.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📬 Contact

- **GitHub**: [@ingvarchernov](https://github.com/ingvarchernov)
- **Issues**: [Report bugs](https://github.com/ingvarchernov/data/issues)

---

**Built with ❤️ for the crypto trading community**
