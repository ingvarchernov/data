# Pattern Chart Visualization

## 📊 Що це?

Автоматична візуалізація технічних паттернів на криптовалютних графіках.

## ✨ Особливості

### Графік містить:

1. **Price Chart (верхня панель)**
   - 📈 Candlestick chart (свічковий графік)
     - 🟢 Зелені свічки - зростання
     - 🔴 Червоні свічки - падіння
   - 📊 Ковзні середні (Moving Averages)
     - EMA 9 (cyan/блакитний)
     - EMA 21 (magenta/фіолетовий)  
     - EMA 50 (orange/помаранчевий)
   - 🎯 **Pattern Marker** (позначка паттерну)
     - 🟢 Зелена стрілка вгору ↑ - LONG паттерн
     - 🔴 Червона стрілка вниз ↓ - SHORT паттерн
   - 📦 **Pattern Zone** (зона паттерну)
     - Пунктирна рамка навколо останніх 10 свічок
     - Колір відповідає напрямку паттерну
   - ℹ️ **Info Box** (інформаційний блок)
     - Symbol, Timeframe
     - Pattern Type, Direction
     - Strength, Confidence
     - Price, Timestamp

2. **Volume Chart (середня панель)**
   - Стовпчиковий графік об'єму торгів
   - Колір відповідає напрямку свічки

3. **RSI Chart (нижня панель)**
   - RSI індикатор (0-100)
   - 🔴 Overbought level (70) - перекупленість
   - 🟢 Oversold level (30) - перепроданість
   - Заливка зон (зелена=бичача, червона=ведмежа)

## 🚀 Використання

### Автоматична генерація (з MTF сканера):

```python
from multi_timeframe_scanner import MultiTimeframeScanner
from pattern_chart_visualizer import visualize_top_patterns

# Сканування
scanner = MultiTimeframeScanner()
symbols = await scanner.get_top_symbols(limit=20)
results = await scanner.batch_scan(symbols)

# Візуалізація топ 5 сигналів
charts = await visualize_top_patterns(scanner, results, top_n=5)
```

### Ручна генерація одного графіку:

```python
from pattern_chart_visualizer import PatternVisualizer

visualizer = PatternVisualizer(output_dir='charts')

# df - DataFrame з OHLCV + indicators (ema9, ema21, ema50, rsi)
# pattern - Pattern object з multi_timeframe_scanner

filepath = visualizer.plot_pattern(
    df=df,
    pattern=pattern,
    symbol='BTCUSDT',
    timeframe='1h',
    show_indicators=True,
    save=True
)
```

## 📁 Структура файлів

Графіки зберігаються в директорії `charts/`:

```
charts/
├── BTCUSDT_1h_Double_Top_20251110_191145.png
├── ETHUSDT_1d_Bullish_Engulfing_20251110_120000.png
└── XRPUSDT_4h_Head_and_Shoulders_20251110_150000.png
```

**Формат назви файлу:**
```
{SYMBOL}_{TIMEFRAME}_{PATTERN_TYPE}_{TIMESTAMP}.png
```

## 📊 Приклад виводу

```
2025-11-10 19:11:43 - INFO - 📊 GENERATING CHARTS
2025-11-10 19:11:43 - INFO - 📊 Generating charts for top 5 MTF signals...
2025-11-10 19:11:43 - INFO - 
1. BTCUSDT (MTF Score: 75.7)
2025-11-10 19:11:44 - INFO - 📊 Chart saved: charts/BTCUSDT_4h_Double_Top_20251110_191143.png
2025-11-10 19:11:44 - INFO -   ✅ [4h] Double Top - BTCUSDT_4h_Double_Top_20251110_191143.png
2025-11-10 19:11:46 - INFO - 📊 Chart saved: charts/BTCUSDT_1h_Double_Top_20251110_191145.png
2025-11-10 19:11:46 - INFO -   ✅ [1h] Double Top - BTCUSDT_1h_Double_Top_20251110_191145.png
...
2025-11-10 19:12:12 - INFO - ✅ Generated 17 charts in charts/
```

## 🎨 Кастомізація

### Зміна кольорів:

Редагуйте `pattern_chart_visualizer.py`:

```python
self.colors = {
    'LONG': '#00ff00',      # Зелений для LONG
    'SHORT': '#ff0000',     # Червоний для SHORT
    'NEUTRAL': '#ffff00',   # Жовтий для NEUTRAL
    'candle_up': '#00ff88',
    'candle_down': '#ff4444',
    'ema9': '#00ffff',
    'ema21': '#ff00ff',
    'ema50': '#ffaa00',
}
```

### Зміна розміру графіку:

```python
fig = plt.figure(figsize=(16, 10))  # Змініть (width, height)
```

### Зміна DPI (якість):

```python
plt.savefig(filepath, dpi=150)  # Змініть 150 на 200 для вищої якості
```

## 🧪 Тестування

Швидкий тест візуалізації:

```bash
cd /home/ihor/data_proj/data
source venv/bin/activate
python test_visualization.py
```

Результат:
- ✅ Сканує 20 топових символів
- ✅ Генерує графіки для топ 5 MTF сигналів
- ✅ Зберігає в `charts/`

## 🔧 Вимоги

```bash
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
```

Встановлення:
```bash
pip install matplotlib pandas numpy
```

## 📝 Формат даних

DataFrame повинен містити:

**Обов'язкові колонки:**
- `open`, `high`, `low`, `close`, `volume`
- `timestamp` або індекс з timestamp

**Опціональні (для індикаторів):**
- `ema9`, `ema21`, `ema50`
- `rsi`

Pattern object повинен мати:
- `pattern_type` (PatternType enum)
- `direction` (str: 'LONG', 'SHORT', 'NEUTRAL')
- `confidence` (float: 0-100)
- `price` (float)
- `strength` (PatternStrength enum)
- `timestamp` (datetime)

## 🐛 Troubleshooting

### Warning: Glyph missing from font

```
UserWarning: Glyph 128308 (\N{LARGE RED CIRCLE}) missing from font(s) DejaVu Sans Mono.
```

**Рішення:** Не критично, емоджі в інфо-блоку не відображаються. Графік малюється коректно.

### Warning: tight_layout not compatible

```
UserWarning: This figure includes Axes that are not compatible with tight_layout
```

**Рішення:** Не критично, layout може мати мінімальні недоліки. Використовуйте `bbox_inches='tight'` при savefig.

### No charts generated

1. Перевірте matplotlib:
   ```bash
   python -c "import matplotlib; print(matplotlib.__version__)"
   ```

2. Перевірте права на запис:
   ```bash
   ls -la charts/
   ```

3. Перевірте логи:
   ```bash
   python test_visualization.py 2>&1 | grep -i error
   ```

## 💡 Поради

1. **Для web-інтерфейсу**: використовуйте Plotly замість Matplotlib для інтерактивних графіків

2. **Для batch generation**: обмежте кількість графіків (top_n=5-10) щоб не перевантажити систему

3. **Для архівації**: зберігайте графіки з timestamp в назві для історії

4. **Для аналізу**: відкривайте графіки поруч з різних таймфреймів для MTF confluence

## 📈 Інтеграція з PostgreSQL

Після створення графіку можна зберегти snapshot в БД:

```python
from database import MTFDatabase

db = MTFDatabase()

# Зберегти snapshot для відтворення графіку
snapshot_id = db.save_chart_snapshot(
    symbol='BTCUSDT',
    timeframe='1h',
    pattern_id=pattern_id,
    candles_data=df[['open', 'high', 'low', 'close', 'volume']].to_dict('records'),
    indicators_data={
        'ema9': df['ema9'].tolist(),
        'ema21': df['ema21'].tolist(),
        'rsi': df['rsi'].tolist()
    },
    pattern_coordinates={
        'x': len(df) - 1,
        'y': pattern.price
    }
)
```

## 🔮 Майбутні покращення

- [ ] Інтерактивні графіки (Plotly)
- [ ] Web-інтерфейс для перегляду
- [ ] Автоматичний upload в Telegram
- [ ] Анімація формування паттерну
- [ ] Порівняння декількох таймфреймів на одному графіку
- [ ] Додаткові індикатори (MACD, Bollinger Bands)
- [ ] Fibonacci levels
- [ ] Support/Resistance zones

---

**Створено:** 2025-11-10  
**Версія:** 1.0  
**Статус:** ✅ Працює
