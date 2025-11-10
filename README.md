# 🚀 Crypto Pattern Trading System v2.0

Професійна система для виявлення технічних паттернів на криптовалютному ринку з Multi-Timeframe аналізом та візуалізацією.

## ✨ Ключові можливості

### 🎯 Multi-Timeframe Pattern Detection
- **5 таймфреймів одночасно**: 1d, 4h, 1h, 30m, 15m
- **150+ символів**: топові криптопари по об'єму
- **Швидкість**: 3.3 символи/сек (46 сек для повного скану)
- **MTF Score**: зважена оцінка паттернів через всі таймфрейми
- **Confluence Analysis**: відсоток згоди між таймфреймами

### 📊 Advanced Pattern Recognition
- **Rust-powered detection**: 50x швидше за Python
- **Multiple pattern types**:
  - Candlestick patterns (Engulfing, Hammer, Doji, etc.)
  - Chart patterns (Head & Shoulders, Double Top/Bottom)
  - Trend patterns (Support/Resistance breaks)
  - Breakout patterns
- **Volatility filtering**: відсіювання низьколіквідних ринків
- **Confidence scoring**: 0-100% для кожного паттерну

### 📈 Interactive Chart Visualization
- **Candlestick charts** з кольоровими свічками
- **Technical indicators**: EMA9, EMA21, EMA50, RSI
- **Pattern markers**: візуальні позначки де паттерн виявлено
- **Pattern zones**: рамки навколо зони формування
- **Multi-panel layout**: ціна + об'єм + RSI
- **Auto-save**: графіки зберігаються автоматично

### 🗄️ PostgreSQL Database
- **4 таблиці**: patterns, pattern_stats, mtf_signals, chart_snapshots
- **3 VIEW**: recent_patterns, top_recurring, top_mtf_signals
- **Indexes**: швидкі запити по symbol, timeframe, timestamp
- **ACID transactions**: надійність даних
- **JSONB support**: гнучке зберігання метаданих

## 🏗️ Архітектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Timeframe Scanner                   │
│  - Binance API (top 150 symbols)                            │
│  - Parallel batch processing (20 symbols)                   │
│  - 5 timeframes × 150 symbols = 750 analyses                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pattern Detection Layer                    │
│  - Rust Pattern Detector (50x faster)                       │
│  - Volatility Filter (min score 50)                         │
│  - Confidence calculation                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐       ┌─────────────────────┐
│  PostgreSQL DB    │       │  Visualization      │
│  - Patterns       │       │  - Charts           │
│  - Statistics     │       │  - Indicators       │
│  - MTF Signals    │       │  - Pattern markers  │
│  - Chart data     │       │  - Auto-save        │
└───────────────────┘       └─────────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
                ┌──────────────┐
                │  Telegram    │
                │  Alerts      │
                └──────────────┘
```

## 📦 Компоненти

### Core Files

**Pattern Detection:**
- `pattern_scanner.py` - головний сканер паттернів
- `pattern_detector/` - Rust модуль (50x швидше)
- `volatility_filter.py` - фільтр волатильності

**Multi-Timeframe Analysis:**
- `multi_timeframe_scanner.py` - MTF сканер (5 TF)
- `daily_scanner.py` - автоматичний щоденний скан

**Visualization:**
- `pattern_chart_visualizer.py` - генерація графіків
- `charts/` - збережені графіки

**Database:**
- `database/mtf_db.py` - PostgreSQL клас
- `schema_mtf_patterns.sql` - SQL схема

**Testing:**
- `test_mtf_quick.py` - швидкий MTF тест
- `test_visualization.py` - тест візуалізації

### Utilities

- `unified_binance_loader.py` - універсальний Binance API
- `config.py` - конфігурація
- `telegram_bot.py` - Telegram інтеграція

## 🚀 Швидкий старт

### 1. Встановлення

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/crypto-pattern-trading.git
cd crypto-pattern-trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Конфігурація

Створіть `.env` файл:

```env
# Binance API (опціонально для публічних даних)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_patterns
POSTGRES_USER=trader
POSTGRES_PASSWORD=trading123
```

### 3. Ініціалізація PostgreSQL

```bash
# Створити базу даних та таблиці
python -c "from database.mtf_db import init_database; init_database()"

# Мігрувати дані з JSON (якщо є)
python -c "from database.mtf_db import migrate_from_json; migrate_from_json()"
```

### 4. Запуск

**MTF сканування (150 символів):**
```bash
python multi_timeframe_scanner.py
```

**Швидкий тест (20 символів):**
```bash
python test_mtf_quick.py
```

**Візуалізація:**
```bash
python test_visualization.py
# Графіки збережуться в charts/
```

**Daily Scanner (systemd):**
```bash
sudo systemctl start daily-scanner
sudo systemctl status daily-scanner
```

## 📊 Приклад результатів

### MTF Scan Output:
```
🚀 Starting batch scan: 150 symbols, batch_size=20
⏱️ Scan completed in 46.2 seconds
📊 Symbols: 150 | Timeframes: 750 | Patterns: 2315

TOP 10 MTF SIGNALS:
1. BTCUSDT    | MTF Score: 75.7 | Patterns: 20
   Timeframes: 4/5 | Direction: SHORT (100% confluence)
   
2. ETHUSDT    | MTF Score: 75.0 | Patterns: 15
   Timeframes: 3/5 | Direction: SHORT (100% confluence)
```

### Generated Charts:
```
charts/
├── BTCUSDT_4h_Double_Top_20251110_191143.png
├── BTCUSDT_1h_Double_Top_20251110_191145.png
├── ETHUSDT_1d_Double_Top_20251110_191150.png
└── ... (17 графіків всього)
```

## 📈 Статистика

**Тестовий прогон (150 символів):**
- ⏱️ **Час**: 46 секунд
- 📊 **Таймфреймів**: 750 (150 × 5)
- 🎯 **Паттернів**: 2315
- 📞 **API calls**: 750
- 💾 **Cache hits**: 0% (перший запуск)

**Швидкий тест (20 символів):**
- ⏱️ **Час**: 10.8 секунд
- 🎯 **Паттернів**: 465
- ✅ **100% confluence** на топових сигналах

## 🔧 Технології

- **Python 3.12+**
- **Rust** (pattern_detector) - maturin compiled
- **PostgreSQL** - pattern storage
- **Matplotlib** - chart visualization
- **python-binance** - market data
- **asyncio** - parallel processing
- **Telegram Bot API** - alerts

## 📚 Документація

- [MTF Scanner README](MTF_SCANNER_README.md) - Multi-timeframe scanner
- [System Summary](MTF_SYSTEM_SUMMARY.md) - Повний огляд системи
- [System Changes](SYSTEM_CHANGES_SUMMARY.md) - Що було додано
- [Charts README](charts/README.md) - Візуалізація графіків

## 🎯 Use Cases

1. **Pattern Discovery**: знайти recurring patterns на багатьох парах
2. **Multi-Timeframe Confirmation**: confluence між таймфреймами
3. **Visual Analysis**: графіки з позначеними паттернами
4. **Statistical Analysis**: PostgreSQL для аналітики
5. **Telegram Alerts**: автоматичні сповіщення про сигнали

## ⚙️ Конфігурація

### Timeframe Weights (MTF Score):
```python
TF_WEIGHTS = {
    '1d': 5.0,   # Найбільша вага
    '4h': 3.0,
    '1h': 2.0,
    '30m': 1.5,
    '15m': 1.0   # Найменша вага
}
```

### Signal Thresholds:
```python
# Recurring pattern
MIN_OCCURRENCES = 5
MIN_CONFIDENCE = 65.0

# MTF signal
MIN_MTF_SCORE = 70.0
MIN_CONFLUENCE = 60.0
MIN_TIMEFRAMES = 3
```

## 🐛 Troubleshooting

### Matplotlib warnings
```
UserWarning: Glyph missing from font
```
**Рішення:** Не критично, графіки малюються коректно

### PostgreSQL connection error
```
psycopg2.OperationalError: could not connect
```
**Рішення:** 
```bash
sudo systemctl start postgresql
psql -U postgres -c "CREATE USER trader WITH PASSWORD 'trading123';"
```

### Rust compilation error
```bash
# Rebuild Rust module
cd pattern_detector
maturin develop --release
```

## 🔮 Roadmap

- [ ] Web interface (Flask/FastAPI)
- [ ] Real-time WebSocket scanning
- [ ] Backtesting framework
- [ ] ML-based confidence improvement
- [ ] Risk/Reward calculator
- [ ] Auto-trading integration
- [ ] Plotly interactive charts
- [ ] Pattern success rate tracking

## 📝 License

Private repository - All rights reserved

## 👤 Author

**Ihor Chernov**

## 🙏 Acknowledgments

- Binance API
- Rust pattern_detector
- Python-binance library
- PostgreSQL community

---

**Created:** 2025-11-10  
**Version:** 2.0.0  
**Status:** ✅ Production Ready
