# 📊 Analytics - Аналіз та Візуалізація Торгової Системи

Цей каталог містить всі інструменти для аналізу та візуалізації моделей і торгових сигналів.

## 📁 Структура

```
analytics/
├── README.md                      # Ця документація
├── analyze_classification.py      # Аналіз класифікаційної моделі (3 класи)
├── analyze_regression.py          # Аналіз регресійної моделі (безперервні прогнози)
├── visualize_signals.py           # Візуалізація торгових сигналів
├── test_signal_strategies.py      # Тестування стратегій (оригінал vs інверсія)
├── monitor_training.py            # Моніторинг процесу тренування в реальному часі
├── graphics/                      # Згенеровані графіки та дані
│   ├── *.png                     # PNG візуалізації
│   └── csv/                      # CSV файли з результатами
│       └── classification_analysis_*.csv
└── logs/                         # Логи тренувань
    ├── train_output*.log
    └── training_log*.txt
```

## 🚀 Основні Скрипти

### 1. analyze_classification.py
**Призначення:** Аналіз класифікаційної моделі BTC з генерацією торгових сигналів

**Тип моделі:** Класифікація (3 класи: DOWN, NEUTRAL, UP)

**Використання:**
```bash
python analytics/analyze_classification.py --symbol BTCUSDT --days 14 --recent 30
```

**Параметри:**
- `--symbol` - Торговий символ (default: BTCUSDT)
- `--model` - Шлях до моделі (optional)
- `--days` - Днів історичних даних (default: 14)
- `--recent` - Скільки останніх сигналів показати (default: 25)

**Виводить:**
- Розподіл прогнозів (DOWN/NEUTRAL/UP)
- Останні N сигналів з рекомендаціями
- Поточний торговий сигнал
- Торгову стратегію (Target/Stop-loss)
- CSV файл у `graphics/csv/`

**Особливості:**
- ✅ **Автоматична інверсія UP↔DOWN** (модель була навчена інвертовано)
- 📊 **Помірно-агресивний режим** - баланс між сигналами та HOLD
- 🎯 Використовує пороги: DOWN<-0.7%, NEUTRAL=-0.7%..+0.7%, UP>+0.7%

---

### 2. analyze_regression.py
**Призначення:** Аналіз регресійної моделі з безперервними прогнозами зміни ціни

**Тип моделі:** Регресія (одне значення - відносна зміна ціни)

**Використання:**
```bash
python analytics/analyze_regression.py --symbol BTCUSDT --days 7 --threshold 0.005
```

**Параметри:**
- `--symbol` - Торговий символ (default: BTCUSDT)
- `--model-dir` - Папка з моделлю (default: models/optimized_BTC)
- `--days` - Скільки днів даних завантажити (default: 7)
- `--threshold` - Поріг для сигналів (default: 0.005 = 0.5%)
- `--recent` - Скільки останніх сигналів показати (default: 20)

**Виводить:**
- Прогнозовану відносну зміну ціни (%)
- Торгові сигнали: BUY/SELL/HOLD
- Силу сигналу (0.0-3.0)
- Статистику розподілу сигналів
- CSV файл з результатами

**Відмінності від classification:**
- ❌ **Без інверсії** - regression моделі працюють коректно
- 📈 **Прогноз конкретної зміни** - наприклад +1.23% або -0.87%
- 🎲 **Простіша логіка** - порівняння з порогом ±0.5%

**Сигнали:**
- `predicted_change > +0.5%` → BUY 🟢
- `predicted_change < -0.5%` → SELL 🔴
- `-0.5% ≤ change ≤ +0.5%` → HOLD ⚪

---

### 3. visualize_signals.py
**Призначення:** Створення графічної візуалізації торгових сигналів на ціновому графіку

**Використання:**
```bash
python analytics/visualize_signals.py --symbol BTCUSDT --days 14
```

**Параметри:**
- `--symbol` - Торговий символ (default: BTCUSDT)
- `--days` - Період даних (default: 14)
- `--output` - Файл для збереження (optional)

**Створює 4 підграфіки:**
1. **Ціна + Сигнали** - графік BTC з позначками BUY/SELL
   - 🟢 Зелені трикутники вгору = BUY
   - 🔴 Червоні трикутники вниз = SELL
   - 💪 Великі з обводкою = STRONG сигнали
   
2. **Різниця UP-DOWN** - барчарт різниці ймовірностей
   
3. **Розподіл ймовірностей** - area chart з трьома класами
   
4. **Статистика** - інфобокс з метриками

**Результат:** PNG файл у `graphics/trading_signals_*.png`

---

### 4. test_signal_strategies.py
**Призначення:** Порівняння оригінальних vs інвертованих сигналів моделі

**Використання:**
```bash
python analytics/test_signal_strategies.py
```

**Тестує:**
- Оригінальну стратегію (як модель predict)
- Інвертовану стратегію (UP↔DOWN)

**Метрики:**
- Точність UP/DOWN сигналів
- Загальна accuracy
- Прибутковість ($)
- Середній прибуток на сигнал

**Приклад виводу:**
```
📊 СТРАТЕГІЯ: Інвертована (UP↔DOWN)
📈 UP сигналів: 6
   ✅ Правильно: 4 (66.7%)
📉 DOWN сигналів: 72
   ✅ Правильно: 39 (54.2%)
🎯 Загальна точність: 29.1%
💰 Сумарний прибуток: $299.26

✅ РЕКОМЕНДАЦІЯ: Інвертувати сигнали!
```

---

### 5. monitor_training.py
**Призначення:** Моніторинг процесу тренування моделі в реальному часі

**Використання:**
```bash
python analytics/monitor_training.py --log train_output.log --interval 10
```

**Параметри:**
- `--log` - Файл логу для моніторингу (default: train_output.log)
- `--interval` - Інтервал оновлення в секундах (default: 10)

**Відслідковує:**
- Прогрес epochs (progress bar)
- Val accuracy та val loss по epoch
- Дельта метрик (зростання/падіння)
- Найкраща epoch
- Checkpoint збереження
- Early stopping

**Використання разом з тренуванням:**
```bash
# Термінал 1: Запуск тренування
python train_classification.py > train_output.log 2>&1 &

# Термінал 2: Моніторинг
python analytics/monitor_training.py --log train_output.log
```

---

## 📈 Типовий Workflow

### Аналіз Класифікаційної Моделі
```bash
# 1. Запуск аналізу classification моделі
python analytics/analyze_classification.py --symbol BTCUSDT --days 30

# 2. Створення візуалізації
python analytics/visualize_signals.py --symbol BTCUSDT --days 30

# 3. Перевірка результатів
ls -lh analytics/graphics/*.png
cat analytics/graphics/csv/classification_analysis_*.csv | tail -20
```

### Аналіз Регресійної Моделі
```bash
# 1. Запуск аналізу regression моделі
python analytics/analyze_regression.py --symbol BTCUSDT --days 7 --threshold 0.005

# 2. Більш агресивні сигнали (поріг 0.3%)
python analytics/analyze_regression.py --threshold 0.003

# 3. Перевірка результатів
cat analysis_BTCUSDT_*.csv | tail -30
```

### Тестування Стратегії
```bash
# Порівняння оригінальних vs інвертованих сигналів
python analytics/test_signal_strategies.py
```

### Моніторинг Нового Тренування
```bash
# Запуск тренування у фоні
python train_classification.py > train_output.log 2>&1 &

# Моніторинг у реальному часі
python analytics/monitor_training.py --log train_output.log --interval 5
```

---

## 🔧 Налаштування

### Зміна Режиму Інтерпретації Сигналів

У `analyze_classification.py` метод `generate_signals()`:

**Поточний режим:** Помірно-агресивний
- NEUTRAL при різниці UP-DOWN < 1% або впевненості NEUTRAL > 55%
- UP/DOWN при різниці > 4% або ймовірності > 35%

Можна змінити пороги для більш агресивної/консервативної торгівлі.

### Зміна Порогів Класифікації

У `training/models/classification_trainer.py`:
```python
'down_threshold': -0.007,  # -0.7%
'up_threshold': 0.007,     # +0.7%
```

---

## 📊 Результати

### Поточна Модель (classification_BTC)
- **Параметрів:** 1,502,467
- **Архітектура:** Bidirectional LSTM + Multi-Head Attention
- **Input:** (120, 41) - 120 timesteps, 41 features
- **Output:** 3 класи (DOWN, NEUTRAL, UP)
- **Навчено:** 181+ epochs

### Метрики (з інверсією UP↔DOWN)
- **Загальна точність:** ~29%
- **UP accuracy:** 66.7%
- **DOWN accuracy:** 54.2%
- **Прибутковість:** +$299 на 149 сигналах

### Розподіл Сигналів (помірно-агресивний)
- DOWN: 55%
- NEUTRAL: 40%
- UP: 5%

---

## ⚠️ Важливі Примітки

### Інверсія Сигналів
Модель була навчена з **інвертованими лейблами**. Тому:
- ✅ У `analyze_classification.py` автоматично міняємо UP↔DOWN
- ❌ Без інверсії: точність 23%, збитки -$299
- ✅ З інверсією: точність 29%, прибуток +$299

### Дисбаланс Класів
- Ринок BTC у флеті → 89.6% даних класифікується як NEUTRAL
- Class weights 8.0 для UP/DOWN, 0.35 для NEUTRAL
- Модель схильна predict NEUTRAL (легший шлях)

### Рекомендації
- ✅ Використовуйте помірно-агресивний режим
- ⚠️ Точність 29% недостатня для продакшену (потрібно 55%+)
- 🔄 Розгляньте перенавчання або бінарну класифікацію (UP/DOWN only)

---

## 🛠️ Технічні Деталі

### Залежності
- TensorFlow 2.20.0 (CPU-only через CUDA_VISIBLE_DEVICES="")
- Rust-accelerated indicators (fast_indicators/)
- UnifiedBinanceLoader для даних
- Matplotlib для візуалізації

### Features (41 індикаторів)
- Технічні: RSI, MACD, Bollinger, ATR, Stochastic
- Price-based: returns, log_returns, momentum
- Volume-based: volume indicators
- Rust-прискорені розрахунки

---

## 📝 Історія Змін

### 24.10.2025 - Виявлення та виправлення інверсії
- ❌ Виявлено: модель давала інвертовані сигнали
- ✅ Виправлено: додано інверсію UP↔DOWN в predict()
- 📈 Результат: точність +5.5%, прибутковість +$598

### 23.10.2025 - Оптимізація генерації сигналів
- ❌ Проблема: 100% NEUTRAL (занадто консервативно)
- ✅ Рішення: помірно-агресивна логіка
- 📊 Результат: 55% DOWN, 40% NEUTRAL, 5% UP

### 21.10.2025 - Створення класифікаційної моделі
- 🏗️ Архітектура: Bi-LSTM + Attention
- 📚 Навчання: 181 epochs на 8760 записах (365 днів)
- 💾 Збережено: models/classification_BTC/

---

## 🔗 Пов'язані Файли

- `train_classification.py` - скрипт тренування
- `training/models/classification_trainer.py` - тренер моделі
- `unified_binance_loader.py` - завантаження даних
- `fast_indicators/` - Rust індикатори
- `models/classification_BTC/` - збережені моделі

---

**Останнє оновлення:** 24.10.2025
