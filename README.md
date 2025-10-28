# 🤖 Crypto Trading ML System# 🤖 Crypto Trading ML System# 📊 Crypto Trading System - Модульна Архітектура



**Random Forest Trend Classifier** для прогнозування напрямку крипто-ринку.



> **Статус**: ✅ Random Forest Implementation Complete (27.10.2025)**Random Forest Trend Classifier** для прогнозування напрямку крипто-ринку.> **Статус**: ✅ Рефакторинг Phases 1-6 Complete (23.10.2025)



## 🎯 Результати



- **BTCUSDT**: 81.15% accuracy ✅## 🎯 РезультатиСистема автоматизованого трейдингу криптовалютою з ML-прогнозуванням, технічним аналізом та повним ризик-менеджментом.

- **Timeframe**: 4h (оптимальний баланс signal/noise)

- **Метод**: Random Forest (50 estimators, max_depth=5)

- **Features**: 82 технічні індикатори (через Rust)

- **Час тренування**: 3-7 секунд на модель- **BTCUSDT**: 81.15% accuracy ✅---



## 📁 Структура проекту- **Timeframe**: 4h (оптимальний баланс signal/noise)



```- **Метод**: Random Forest (50 estimators, max_depth=5)## 🎯 Основні можливості

data/

├── training/- **Features**: 82 технічні індикатори (через Rust)

│   ├── simple_trend_classifier.py  # Основна модель Random Forest

│   ├── batch_train_rf.py          # Batch тренування для всіх валют- 🤖 **ML Прогнозування**: LSTM, Transformer моделі з 85%+ точністю

│   ├── rust_features.py            # Rust indicators wrapper

│   └── __init__.py## 📁 Структура проекту- 📈 **Технічний аналіз**: 70+ індикаторів з Rust acceleration

├── fast_indicators/                # Rust-based technical indicators (264MB)

│   ├── src/- 🎯 **Стратегії**: Swing Trading, Scalping, Day Trading

│   └── Cargo.toml

├── models/```- ⚡ **Live Trading**: Real-time торгівля з ризик-менеджментом

│   └── simple_trend_*/            # Збережені Random Forest моделі

├── train_models.py                # 🚀 Швидке тренування всіх моделейdata/- 💾 **База даних**: PostgreSQL + Redis caching

├── train_single.py                # 🎯 Тренування однієї монети

└── archive/                       # Старий GRU/TensorFlow код (2.5MB)├── training/- 📊 **Моніторинг**: TensorBoard, Telegram alerts, Performance metrics

```

│   ├── simple_trend_classifier.py  # Основна модель Random Forest

## 🚀 Швидкий старт

│   ├── batch_train_rf.py          # Batch тренування для всіх валют---

### 1. Тренування всіх моделей

│   ├── rust_features.py            # Rust indicators wrapper

```bash

python train_models.py│   └── __init__.py## 📁 Модульна структура (NEW!)

```

├── fast_indicators/                # Rust-based technical indicators

**Результат:**

- Автоматичне тренування для BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, XRPUSDT│   ├── src/Проект повністю рефакторовано на модульну архітектуру:

- Час: ~30-50 секунд для всіх моделей

- Моделі зберігаються в `models/simple_trend_{symbol}/`│   └── Cargo.toml



### 2. Тренування однієї монети├── models/### `training/` - Тренування моделей



```bash│   └── simple_trend_*/            # Збережені моделі```python

# За замовчуванням: 4h timeframe, 730 днів історії

python train_single.py BTCUSDT├── intelligent_sys/               # Data fetching & processingfrom training import BaseModelTrainer, FeatureEngineer



# Кастомні параметри├── database/                      # PostgreSQL schemasfrom training.models import OptimizedTrainer, AdvancedTrainer

python train_single.py ETHUSDT --timeframe 1h --days 365

```├── archive/                       # Старі файли (GRU, TensorFlow, etc)



### 3. Швидкий тест (3 монети)├── unified_binance_loader.py     # Binance API wrapper# Швидкий старт



```bash├── check_gpu.py                  # GPU utilitiestrainer = OptimizedTrainer(symbol='BTCUSDT', interval='1h')

python test_batch_small.py

```└── main.py                       # Старий entry point (не використовується)results = await trainer.train(days=365)



## 📊 Технічні деталі``````



### Random Forest Configuration



```python## 🚀 Використання### `optimized/indicators/` - Технічні індикатори

RandomForestClassifier(

    n_estimators=50,      # 50 дерев```python

    max_depth=5,          # Обмежена глибина (проти overfitting)

    min_samples_split=50, # Мінімум зразків для split### 1. Тренування однієї моделіfrom optimized.indicators import calculate_all_indicators, OptimizedIndicatorCalculator

    class_weight='balanced',  # Баланс класів

    random_state=42

)

``````bash# Batch розрахунок всіх індикаторів (Rust + async)



### Binary Classificationpython training/simple_trend_classifier.py --symbol BTCUSDT --timeframe 4h --days 730calculator = OptimizedIndicatorCalculator(use_async=True, n_workers=4)



- **UP (1)**: Ціна зросте ≥+1.5% протягом 3 періодів (12 годин для 4h)```indicators = await calculator.calculate_all_indicators_batch(df)

- **DOWN (0)**: Інакше

```

### Features (82 індикатори через Rust)

### 2. Batch тренування всіх валют

**Trend:**

- SMA: 10, 20, 50, 100, 200### `optimized/model/` - ML Компоненти

- EMA: 12, 20, 26, 50, 100, 200

- SMA/EMA ratios```bash```python



**Momentum:**python training/batch_train_rf.pyfrom optimized.model import mape, directional_accuracy

- RSI: 7, 14, 28

- MACD, MACD Signal, MACD Histogram```from optimized.model import DatabaseHistoryCallback, DenormalizedMetricsCallback

- ROC: 5, 10, 20

- Stochastic Oscillatorfrom optimized.model import TransformerBlock, PositionalEncoding



**Volatility:**### 3. Валюти```

- ATR: 7, 14, 21

- Standard Deviation: 5, 10, 20, 30

- Bollinger Bands: 20, 50

- Historical Volatility: 10, 20, 30- BTCUSDT### `optimized/database/` - База даних



**Volume:**- ETHUSDT  ```python

- OBV (On-Balance Volume)

- VWAP (Volume Weighted Average Price)- BNBUSDTfrom optimized.database import DatabaseConnection, CacheManager

- Volume ratios & momentum

- SOLUSDT

**Price Action:**

- Returns, Log Returns- ADAUSDT# Connection pooling + Redis cache

- Distance from means

- Candle patterns (body, wicks, ratios)- DOGEUSDTdb = DatabaseConnection(pool_size=20)



## 🎯 Порівняння з GRU (старий підхід)- XRPUSDTcache = CacheManager(use_redis=True, cache_ttl=3600)



| Метрика | GRU (TensorFlow) | Random Forest |```

|---------|------------------|---------------|

| **Train Accuracy** | 87-90% | 79% |## 🔧 Технічний стек

| **Test Accuracy** | 47-62% ❌ | **81%** ✅ |

| **Training Time** | 1.5-2.5 годин | **3-7 секунд** |---

| **Overfitting** | Катастрофічний | Відсутній |

| **GPU Required** | Так | Ні (CPU-only) |- **ML**: scikit-learn (Random Forest)

| **Parameters** | 338,113 | ~5,000 |

- **Indicators**: Rust (fast_indicators)## 🚀 Швидкий старт

**Висновок:** Random Forest > GRU для короткострокового прогнозування крипто.

- **Data**: python-binance

## 📈 Результати по валютах

- **DB**: PostgreSQL (не обов'язково для RF моделей)### 1. Клонування репозиторію

| Symbol | Test Accuracy | Status |

|--------|--------------|--------|```bash

| BTCUSDT | **81.15%** | ✅ Ready (>70%) |

| BNBUSDT | 68.67% | ⚠️ Needs improvement |## 📊 Features (82 індикатори)git clone https://github.com/ingvarchernov/data.git

| ETHUSDT | 64.83% | ⚠️ Needs improvement |

cd data

**Цільовий показник:** 70%+ accuracy

### Trend Indicators```

## 🔧 Налаштування середовища

- SMA (5, 10, 20, 50, 100, 200)

### 1. Встановлення залежностей

- EMA (9, 12, 21, 26, 50)### 2. Встановлення залежностей

```bash

# Python packages- Price vs MA ratios```bash

pip install -r requirements.txt

python -m venv venv

# Rust indicators (якщо треба перебудувати)

cd fast_indicators### Momentumsource venv/bin/activate  # Linux/Mac

cargo build --release

```- RSI (7, 14, 21, 28)# або venv\Scripts\activate  # Windows



### 2. API ключі (для live trading)- MACD



Створіть `.env` файл:- Momentum (5, 10, 20 periods)pip install -r requirements.txt



```env```

FUTURES_API_KEY=your_binance_api_key

FUTURES_API_SECRET=your_binance_api_secret### Volatility

USE_TESTNET=true  # false для production

```- ATR (14, 21)### 3. Налаштування environment



## 📦 Залежності- Bollinger Bands```bash



**Core:**- Historical Volatilitycp .env.example .env

- `python-binance` - Binance API

- `scikit-learn` - Random Forest# Відредагуйте .env та додайте API keys

- `pandas`, `numpy` - Data processing

- `joblib` - Model serialization### Volume```



**Indicators:**- OBV, VWAP

- `fast_indicators` - Rust library (custom)

- Volume trends & ratios### 4. Тренування моделі

**Optional (для old GRU code в archive/):**

- `tensorflow`, `keras` - Deep learning (не використовується)```bash



## 🗂️ Архів## 🎯 Target# Оптимізований trainer з топ-35 фічами



Старий код (GRU/TensorFlow) переміщено в `archive/`:python training/models/optimized_trainer.py --symbol BTCUSDT --days 365

- `archive/old_training_20251027/` - Старі trainers, batch scripts

- `archive/old_root_files_20251027/` - GPU config, cache system, etc**Binary Classification:**

- `archive/old_analytics_20251027/` - Analytics для GRU моделей

- `archive/old_tests_20251027/` - Старі тести- UP (1): +1.5%+ за 3 періоди (12h для 4h timeframe)# Advanced trainer з Rust індикаторами



**Розмір архіву:** 2.5MB (74 файли)- DOWN (0): іншеpython training/models/advanced_trainer.py --symbol BTCUSDT --days 730 --no-rust



## 🚧 TODO```



### Короткострокові## 📈 Модель



- [ ] Покращити accuracy для ETHUSDT, BNBUSDT до 70%+---

- [ ] Експеримент з різними thresholds (1.5% → 1.0% або 2.0%)

- [ ] Тестування на різних lookback periods (3 → 2 або 5)```python



### ДовгостроковіRandomForestClassifier(## 📊 Результати тренування



- [ ] Інтеграція в live trading систему    n_estimators=50,

- [ ] Ensemble моделей з різних timeframes

- [ ] Automatic retraining scheduler    max_depth=5,Досягнуто **85.78% validation accuracy** на класифікації напрямку руху ціни BTC:

- [ ] Monitoring dashboard

    min_samples_split=50,

## 📝 Changelog

    class_weight='balanced'| Метрика | Значення |

### 27.10.2025 - Random Forest Migration Complete

)|---------|----------|

- ✅ Видалено `main.py` (старий GRU-based)

- ✅ Створено `train_models.py` та `train_single.py````| **Validation Accuracy** | 85.78% |

- ✅ Очищено `training/` від старих файлів

- ✅ Переміщено 74 файли в `archive/`| **Test Accuracy** | 83.67% |

- ✅ Усунуто всі import помилки

- ✅ Оновлено документацію**Чому Random Forest?**| **NEUTRAL Precision** | 91% |



### 26.10.2025 - Перехід на Random Forest- ✅ Не потребує GPU| **NEUTRAL Recall** | 93% |



- ✅ Створено `SimpleTrendClassifier` (Random Forest)- ✅ Швидке тренування (~1-7s)| **Training Time** | 182 epochs |

- ✅ Досягнуто 81.15% accuracy на BTCUSDT

- ✅ Інтеграція Rust indicators- ✅ Немає overfitting (train 76%, test 77%)

- ✅ Batch training система

- ✅ Інтерпретуємо (feature importance)**Модель**: Classification (3 classes: DOWN/NEUTRAL/UP)  

### 20.10.2025 - Відмова від GRU

- ❌ GRU/LSTM overfit на крипто даних (train 89%, test 28%)**Архітектура**: Bidirectional LSTM [256,128,64] + Dense [256,128]  

- ❌ GRU models: 47-51% accuracy (провал)

- 📊 Виявлено критичний overfitting**Dataset**: BTCUSDT 1h, 730 days, ~17,520 records

- 🔄 Рішення: перехід на Random Forest

## 🗂️ Archive

---

---

**License:** MIT  

**Author:** ihor  Старі підходи збережені в `archive/`:

**Last Updated:** 27.10.2025

- GRU/LSTM моделі (TensorFlow)## 🏗️ Архітектура проекту

- Multi-timeframe features

- Regression моделі```

- Analytics для deep learningdata/

├── training/                # Модуль тренування моделей

**Причина відмови від GRU:**│   ├── models/

- Overfitting (train 87-90%, test 28-62%)│   │   ├── optimized_trainer.py    # Топ-35 features

- Повільне тренування (2-6 хв на модель)│   │   └── advanced_trainer.py     # Rust indicators

- Потребує GPU│   ├── base_trainer.py             # BaseModelTrainer

- Accuracy < 70% на більшості валют│   ├── feature_engineering.py      # FeatureEngineer

│   ├── data_loader.py              # Async DataLoader

## 📝 Примітки│   └── utils.py                    # Utilities

│

- **4h timeframe** працює краще за 1h (менше шуму) та 1d (більше даних)├── optimized/               # Оптимізовані компоненти

- **BTC має найкращу predictability** (81% vs 62-65% для altcoins)│   ├── indicators/

- Кожна валюта потребує **окремого тренування**│   │   ├── trend.py                # SMA, EMA, MACD

- Random Forest > Deep Learning для крипто на короткому горизонті│   │   ├── momentum.py             # RSI, Stochastic

│   │   ├── volatility.py           # ATR, Bollinger

## 🔮 Наступні кроки│   │   ├── volume.py               # OBV, VWAP

│   │   └── calculator.py           # Batch processing

1. ✅ Досягнуто 70%+ accuracy на BTC│   ├── model/

2. ⏳ Покращити accuracy для altcoins (60-65% → 70%+)│   │   ├── metrics.py              # MAPE, accuracy

3. ⏳ Інтеграція в live trading│   │   ├── callbacks.py            # DB, metrics

4. ⏳ Ensemble з різних timeframes│   │   └── layers.py               # Transformer

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
