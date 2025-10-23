# 📊 Crypto Trading System - Модульна Архітектура

> **Статус**: ✅ Рефакторинг Phases 1-6 Complete (23.10.2025)

Система автоматизованого трейдингу криптовалютою з ML-прогнозуванням, технічним аналізом та повним ризик-менеджментом.

---

## 🎯 Основні можливості

- 🤖 **ML Прогнозування**: LSTM, Transformer моделі з 85%+ точністю
- 📈 **Технічний аналіз**: 70+ індикаторів з Rust acceleration
- 🎯 **Стратегії**: Swing Trading, Scalping, Day Trading
- ⚡ **Live Trading**: Real-time торгівля з ризик-менеджментом
- 💾 **База даних**: PostgreSQL + Redis caching
- 📊 **Моніторинг**: TensorBoard, Telegram alerts, Performance metrics

---

## 📁 Модульна структура (NEW!)

Проект повністю рефакторовано на модульну архітектуру:

### `training/` - Тренування моделей
```python
from training import BaseModelTrainer, FeatureEngineer
from training.models import OptimizedTrainer, AdvancedTrainer

# Швидкий старт
trainer = OptimizedTrainer(symbol='BTCUSDT', interval='1h')
results = await trainer.train(days=365)
```

### `optimized/indicators/` - Технічні індикатори
```python
from optimized.indicators import calculate_all_indicators, OptimizedIndicatorCalculator

# Batch розрахунок всіх індикаторів (Rust + async)
calculator = OptimizedIndicatorCalculator(use_async=True, n_workers=4)
indicators = await calculator.calculate_all_indicators_batch(df)
```

### `optimized/model/` - ML Компоненти
```python
from optimized.model import mape, directional_accuracy
from optimized.model import DatabaseHistoryCallback, DenormalizedMetricsCallback
from optimized.model import TransformerBlock, PositionalEncoding
```

### `optimized/database/` - База даних
```python
from optimized.database import DatabaseConnection, CacheManager

# Connection pooling + Redis cache
db = DatabaseConnection(pool_size=20)
cache = CacheManager(use_redis=True, cache_ttl=3600)
```

---

## 🚀 Швидкий старт

### 1. Клонування репозиторію
```bash
git clone https://github.com/ingvarchernov/data.git
cd data
```

### 2. Встановлення залежностей
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Налаштування environment
```bash
cp .env.example .env
# Відредагуйте .env та додайте API keys
```

### 4. Тренування моделі
```bash
# Оптимізований trainer з топ-35 фічами
python training/models/optimized_trainer.py --symbol BTCUSDT --days 365

# Advanced trainer з Rust індикаторами
python training/models/advanced_trainer.py --symbol BTCUSDT --days 730 --no-rust
```

---

## 📊 Результати тренування

Досягнуто **85.78% validation accuracy** на класифікації напрямку руху ціни BTC:

| Метрика | Значення |
|---------|----------|
| **Validation Accuracy** | 85.78% |
| **Test Accuracy** | 83.67% |
| **NEUTRAL Precision** | 91% |
| **NEUTRAL Recall** | 93% |
| **Training Time** | 182 epochs |

**Модель**: Classification (3 classes: DOWN/NEUTRAL/UP)  
**Архітектура**: Bidirectional LSTM [256,128,64] + Dense [256,128]  
**Dataset**: BTCUSDT 1h, 730 days, ~17,520 records

---

## 🏗️ Архітектура проекту

```
data/
├── training/                # Модуль тренування моделей
│   ├── models/
│   │   ├── optimized_trainer.py    # Топ-35 features
│   │   └── advanced_trainer.py     # Rust indicators
│   ├── base_trainer.py             # BaseModelTrainer
│   ├── feature_engineering.py      # FeatureEngineer
│   ├── data_loader.py              # Async DataLoader
│   └── utils.py                    # Utilities
│
├── optimized/               # Оптимізовані компоненти
│   ├── indicators/
│   │   ├── trend.py                # SMA, EMA, MACD
│   │   ├── momentum.py             # RSI, Stochastic
│   │   ├── volatility.py           # ATR, Bollinger
│   │   ├── volume.py               # OBV, VWAP
│   │   └── calculator.py           # Batch processing
│   ├── model/
│   │   ├── metrics.py              # MAPE, accuracy
│   │   ├── callbacks.py            # DB, metrics
│   │   └── layers.py               # Transformer
│   └── database/
│       ├── connection.py           # Pooling
│       └── cache.py                # Redis + memory
│
├── strategies/              # Торгові стратегії
│   ├── base/                       # Базові класи
│   ├── swing_trading/              # Swing стратегія
│   ├── scalping/                   # Скальпінг
│   └── day_trading/                # Денна торгівля
│
├── intelligent_sys/         # Інтелектуальна система
│   ├── client_manager.py           # Binance client
│   ├── data_fetcher.py             # Data fetching
│   └── data_processor.py           # Processing
│
├── fast_indicators/         # Rust індикатори (25x швидше)
│   ├── src/
│   └── Cargo.toml
│
├── models/                  # Trained models
│   ├── classification_BTC/         # 85.78% acc model
│   ├── optimized_BTC/
│   └── advanced_BTC/
│
├── logs/                    # Training logs
└── tests/                   # Unit tests
```

---

## 🔧 Технології

### ML & Data Science
- **TensorFlow 2.x** - Deep Learning
- **Keras** - Model API
- **NumPy, Pandas** - Data processing
- **Scikit-learn** - Preprocessing

### Database & Cache
- **PostgreSQL** - Main database
- **Redis** - Distributed cache
- **SQLAlchemy** - ORM + async support

### Trading & APIs
- **Binance API** - Market data & trading
- **python-binance** - API wrapper
- **WebSocket** - Real-time data

### Performance
- **Rust** - Fast indicators (25x speedup)
- **AsyncIO** - Async processing
- **Multiprocessing** - Parallel training

### Monitoring
- **TensorBoard** - Training visualization
- **Telegram Bot** - Alerts
- **Logging** - Structured logs

---

## 📈 Рефакторинг (Oct 2025)

Проведено повний рефакторинг проекту (Phases 1-6):

### До рефакторингу:
- ❌ Монолітні файли 700-2000 рядків
- ❌ Дублювання коду 30-40%
- ❌ Важко тестувати та розширювати
- ❌ 12+ застарілих файлів

### Після рефакторингу:
- ✅ Модульні файли 100-400 рядків
- ✅ Мінімум дублювання (~5%)
- ✅ 20+ окремих тестованих модулів
- ✅ Чітка структура та відповідальності
- ✅ 12 старих файлів архівовано
- ✅ 100% backward compatibility

**Детальний звіт**: [REFACTORING_REPORT.md](REFACTORING_REPORT.md)

---

## 🧪 Тестування

```bash
# Запуск всіх тестів
pytest tests/

# З coverage
pytest --cov=. tests/

# Окремі модулі
pytest tests/training/
pytest tests/optimized/
```

---

## 📚 Документація

- [REFACTORING_REPORT.md](REFACTORING_REPORT.md) - Звіт про рефакторинг
- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - План рефакторингу
- `training/` - Docstrings в кожному модулі
- `optimized/` - API documentation

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Ihor**
- GitHub: [@ingvarchernov](https://github.com/ingvarchernov)

---

## 🙏 Acknowledgments

- Binance API для market data
- TensorFlow team за ML framework
- Rust community за швидкі індикатори
- OpenAI за допомогу в рефакторингу

---

**Last Updated**: 23 жовтня 2025  
**Status**: ✅ Production Ready (Phases 1-4 Complete)
