# 🎯 MASTER CONTROL - Швидкий старт

```bash
# 1. Перевірка моделей
python master_control.py check

# 2. Тренування (10 валют за замовчуванням)
python master_control.py train

# 3. Моніторинг позицій
python master_control.py monitor

# 4. Запуск бота
python master_control.py bot --symbols BTCUSDT ETHUSDT

# 5. ВСЕ РАЗОМ: train → check → monitor → bot
python master_control.py all
```

## 📋 Команди master_control.py

| Команда | Опис | Приклад |
|---------|------|---------|
| `check` | Перевірка наявності моделей | `python master_control.py check` |
| `train` | Тренування ML моделей | `python master_control.py train --symbols BTCUSDT ETHUSDT` |
| `monitor` | Моніторинг позицій на Binance | `python master_control.py monitor` |
| `bot` | Запуск торгового бота | `python master_control.py bot --live` |
| `all` | Повний цикл (train+check+monitor+bot) | `python master_control.py all --force` |

## 🔧 Опції

- `--symbols BTCUSDT ETHUSDT` - вказати конкретні валюти (за замовчуванням топ-10)
- `--days 730` - кількість днів історії для тренування (за замовчуванням 730 = 2 роки)
- `--force` - перетренувати навіть якщо модель існує
- `--live` - увімкнути live trading (ОБЕРЕЖНО! За замовчуванням demo mode)
- `--testnet` - використовувати Binance Testnet (за замовчуванням True)

## 📁 Структура проекту

```
data/
├── master_control.py          # 🎯 ГОЛОВНИЙ СКРИПТ - єдина точка входу
│
├── training/                  # ML Тренування
│   ├── simple_trend_classifier.py  # Random Forest модель
│   ├── batch_train_rf.py           # Batch тренування
│   └── rust_features.py            # Rust індикатори
│
├── models/                    # Збережені моделі
│   ├── simple_trend_BTCUSDT/
│   ├── simple_trend_ETHUSDT/
│   └── ...
│
├── intelligent_sys/           # Data pipeline
│   ├── data_fetcher.py             # Завантаження з Binance
│   ├── data_processor.py           # Обробка даних
│   └── client_manager.py           # API клієнт
│
├── optimized/                 # Оптимізовані компоненти
│   ├── indicators/                 # Технічні індикатори
│   ├── model/                      # ML компоненти
│   └── database/                   # БД утиліти
│
├── strategies/                # Торгові стратегії
│   ├── trend_strategy_4h.py        # Trend following 4h
│   ├── swing_strategy_1h.py        # Swing trading 1h
│   └── base_strategy.py            # Базовий клас
│
├── fast_indicators/           # Rust acceleration (264MB)
│   └── src/lib.rs                  # 82 індикатори
│
├── simple_trading_bot.py      # Головний торговий бот
├── monitor_positions.py       # Моніторинг позицій
├── check_orders.py            # Перевірка ордерів
└── telegram_bot.py            # Telegram сповіщення
```

## 🚀 Типові сценарії

### 1. Перший запуск (тренування + запуск)

```bash
# Натренувати топ-6 валют і запустити бота
python master_control.py all --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT XRPUSDT
```

### 2. Додати нові валюти

```bash
# Тренувати тільки нові валюти
python master_control.py train --symbols MATICUSDT AVAXUSDT DOTUSDT

# Перевірити що додалися
python master_control.py check
```

### 3. Перетренувати існуючі моделі

```bash
# Оновити моделі з свіжими даними
python master_control.py train --force --days 730
```

### 4. Моніторинг без торгівлі

```bash
# Просто подивитись поточні позиції
python master_control.py monitor

# Або використати окремий скрипт
python monitor_positions.py
```

### 5. Запуск бота в демо режимі

```bash
# Аналіз без відкриття позицій
python master_control.py bot --symbols BTCUSDT ETHUSDT
```

### 6. Live trading (ОБЕРЕЖНО!)

```bash
# Реальна торгівля на testnet
python master_control.py bot --symbols BTCUSDT ETHUSDT --live

# Реальна торгівля на mainnet (відключити testnet в .env)
# FUTURES_TESTNET=False
python master_control.py bot --symbols BTCUSDT --live
```

## 📊 Результати тренування

Після тренування ви побачите:

```
================================================================================
📊 ПІДСУМОК ТРЕНУВАННЯ
================================================================================

✅ УСПІШНО НАТРЕНОВАНО: 10
   BTCUSDT      - 80.88%
   ETHUSDT      - 65.83%
   BNBUSDT      - 75.23%
   SOLUSDT      - 72.15%
   ...

================================================================================
Всього: 10 | Успіх: 10 | Пропущено: 0 | Помилки: 0
================================================================================
```

## 🎯 Налаштування

### .env файл

```bash
# Binance API (Testnet)
FUTURES_API_KEY=your_testnet_api_key
FUTURES_API_SECRET=your_testnet_secret
FUTURES_TESTNET=True

# Binance API (Mainnet) - ОБЕРЕЖНО!
# FUTURES_API_KEY=your_mainnet_api_key
# FUTURES_API_SECRET=your_mainnet_secret
# FUTURES_TESTNET=False

# Telegram (опціонально)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Отримати Testnet API ключі

1. Зареєструйтесь: https://testnet.binancefuture.com
2. API Management → Create API
3. Додайте ключі в `.env`

## 🔧 Технічні деталі

### ML Модель

- **Алгоритм**: Random Forest Classifier
- **Features**: 82 технічні індикатори (через Rust)
- **Timeframe**: 4h (оптимальний баланс)
- **Target**: Binary (UP/DOWN) з порогом ±1.5% за 12 годин
- **Accuracy**: 65-85% залежно від валюти

### Індикатори (Rust acceleration)

- **Trend**: SMA, EMA (10-200 періодів)
- **Momentum**: RSI, MACD, ROC, Stochastic
- **Volatility**: ATR, Bollinger Bands, Historical Volatility
- **Volume**: OBV, Volume SMA, Volume trend
- **Custom**: Price distance, Trend strength, Support/Resistance

### Ризик-менеджмент

- Stop Loss: 2-3% від entry
- Take Profit: 4-6% від entry
- Max позицій: 6 одночасно
- Max розмір позиції: 10% балансу
- Trailing stop: динамічний

## 📚 Додаткові скрипти

```bash
# Тренування однієї валюти
python train_single.py BTCUSDT --days 365

# Перевірка GPU
python check_gpu.py

# Перевірка моделей
python check_models.py

# Завантаження даних в БД
python load_all_symbols_data.py --symbols BTCUSDT ETHUSDT --days 730
```

## 🐛 Troubleshooting

### Помилка: "attempt to subtract with overflow"

Rust модуль потребує перекомпіляції:

```bash
cd fast_indicators
maturin develop
cd ..
```

### Помилка: "❌ Жодна модель не завантажена!"

Натренуйте моделі:

```bash
python master_control.py train --symbols BTCUSDT ETHUSDT
```

### Помилка API ключів

Перевірте `.env` файл:

```bash
cat .env  # Linux/Mac
type .env # Windows
```

### Низька точність моделі (<60%)

- Збільшіть історію: `--days 1095` (3 роки)
- Спробуйте інший timeframe
- Додайте більше features
- Перетренуйте: `--force`

## 📈 Моніторинг і логи

Логи зберігаються в:
- `app.log` - головний лог файл
- `logs/` - детальні логи бота

Переглянути в реальному часі:

```bash
# Linux/Mac
tail -f app.log

# Windows PowerShell
Get-Content app.log -Wait
```

## 🎓 Навчання

1. **Спочатку**: Натренуйте моделі на історичних даних
2. **Потім**: Запустіть бота в demo mode (без trading)
3. **Нарешті**: Увімкніть live trading на testnet
4. **Обережно**: Mainnet тільки після успішного testnet досвіду

## ⚠️ Важливо

- 🧪 **Завжди тестуйте на Testnet** перед mainnet
- 💰 **Ризикуйте тільки тими грошима, які готові втратити**
- 📊 **Моніторте позиції** регулярно
- 🔄 **Оновлюйте моделі** кожні 1-2 тижні
- 📈 **Ведіть trading journal** для аналізу

## 📞 Підтримка

Створіть issue на GitHub або напишіть мені.

---

**⚡ Швидкий старт за 30 секунд:**

```bash
# 1. Налаштувати API ключі в .env
# 2. Натренувати і запустити
python master_control.py all
```

**Готово!** 🚀
