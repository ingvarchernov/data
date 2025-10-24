# 🎯 План Очищення Проекту - КОНКРЕТНІ ДІЇ

## 📊 Результати Аналізу

### ✅ Модулі які ВИКОРИСТОВУЮТЬСЯ (залишити):
- `strategies/` - ✅ **17 імпортів** у `live_trading.py`, `strategy_manager.py`, `main.py`
- `trading/` - ✅ **8 імпортів** у різних модулях
- `async_architecture.py` - ✅ використовується в `main.py`
- `main.py` - ✅ основний торговий модуль (910 рядків)

### ⚠️ Deprecated модулі які ПОТРЕБУЮТЬ ЗАМІНИ:

#### 1. **optimized_db.py** - 6 використань
```python
# ФАЙЛИ ДЕ ВИКОРИСТОВУЄТЬСЯ:
- async_architecture.py:650
- main.py:180, 577, 662
- tests/test_optimized_db.py:19, 67
```

#### 2. **optimized_indicators.py** - 1 використання
```python
# ФАЙЛ:
- main.py:323
```

#### 3. **optimized_model.py** - 1 використання
```python
# ФАЙЛ:
- main.py:195
```

---

## 🚀 КРОКИ ВИКОНАННЯ

### Крок 1: Backup ✅
```bash
# Створити git commit
git add .
git commit -m "Backup before cleanup - 2025-10-24"

# АБО tar backup
tar -czf data_backup_2025-10-24.tar.gz .
```

### Крок 2: Видалити 100% deprecated ❌
```bash
# Файл який точно не використовується
rm resume_training.py
```

### Крок 3: Замінити імпорти в main.py 🔄

#### 3.1 Замінити optimized_model (рядок 195)
```python
# БУЛО (main.py:195):
from optimized_model import OptimizedPricePredictionModel

# СТАНЕ:
from training.models.optimized_trainer import OptimizedTrainer
```

#### 3.2 Замінити optimized_indicators (рядок 323)
```python
# БУЛО (main.py:323):
from optimized_indicators import OptimizedIndicatorCalculator

# СТАНЕ:
from optimized.indicators import OptimizedIndicatorCalculator
```

#### 3.3 Замінити optimized_db (рядки 180, 577, 662)
```python
# БУЛО:
from optimized_db import db_manager
from optimized_db import db_manager, save_position

# СТАНЕ:
from optimized.database import DatabaseConnection
db_manager = DatabaseConnection()
```

### Крок 4: Замінити в async_architecture.py 🔄

#### 4.1 Замінити optimized_db (рядок 650)
```python
# БУЛО (async_architecture.py:650):
from optimized_db import db_manager

# СТАНЕ:
from optimized.database import DatabaseConnection
db_manager = DatabaseConnection()
```

### Крок 5: Оновити tests/ 🧪
```bash
# Видалити старі debug тести
rm tests/debug_scaler.py
rm tests/test_scaler_simple.py

# Оновити test_optimized_db.py замінити імпорт
# АБО перемістити в optimized/database/tests/
mkdir -p optimized/database/tests
mv tests/test_optimized_db.py optimized/database/tests/
```

### Крок 6: Видалити deprecated wrappers 🗑️
```bash
# ТІЛЬКИ ПІСЛЯ заміни всіх імпортів!
rm optimized_model.py
rm optimized_db.py  
rm optimized_indicators.py
```

### Крок 7: Архівувати старі файли 📦
```bash
# Стиснути archive/
tar -czf archive_backup_2025-10-24.tar.gz archive/
# Опціонально видалити оригінал:
# rm -rf archive/
```

### Крок 8: Перевірка ✅
```bash
# Перевірити що немає deprecated імпортів
grep -r "from optimized_db import" . --include="*.py"
grep -r "from optimized_model import" . --include="*.py"
grep -r "from optimized_indicators import" . --include="*.py"

# Перевірити що код працює
python -c "from optimized.database import DatabaseConnection; print('✅ OK')"
python -c "from optimized.indicators import OptimizedIndicatorCalculator; print('✅ OK')"
python -c "from training.models.optimized_trainer import OptimizedTrainer; print('✅ OK')"
```

---

## 📝 ДЕТАЛЬНІ ІНСТРУКЦІЇ ПО ФАЙЛАХ

### main.py - 4 заміни

#### Заміна 1 (рядок ~195):
```python
# ЗНАЙТИ:
        from optimized_model import OptimizedPricePredictionModel
        
        if not model_info['path'].exists():
            logger.warning(f"⚠️ Модель не знайдена: {model_info['path']}")
            continue

# ЗАМІНИТИ НА:
        from training.models.optimized_trainer import OptimizedTrainer
        
        if not model_info['path'].exists():
            logger.warning(f"⚠️ Модель не знайдена: {model_info['path']}")
            continue
```

#### Заміна 2 (рядок ~180):
```python
# ЗНАЙТИ:
        from optimized_db import db_manager
        if db_manager:
            db_manager.connect()

# ЗАМІНИТИ НА:
        from optimized.database import DatabaseConnection
        db_manager = DatabaseConnection()
        if db_manager:
            db_manager.connect()
```

#### Заміна 3 (рядок ~323):
```python
# ЗНАЙТИ:
                    from optimized_indicators import OptimizedIndicatorCalculator

# ЗАМІНИТИ НА:
                    from optimized.indicators import OptimizedIndicatorCalculator
```

#### Заміна 4 (рядок ~577):
```python
# ЗНАЙТИ:
            from optimized_db import db_manager, save_position

# ЗАМІНИТИ НА:
            from optimized.database import DatabaseConnection
            db_manager = DatabaseConnection()
```

#### Заміна 5 (рядок ~662):
```python
# ЗНАЙТИ:
                from optimized_db import db_manager

# ЗАМІНИТИ НА:
                from optimized.database import DatabaseConnection
                db_manager = DatabaseConnection()
```

---

### async_architecture.py - 1 заміна

#### Заміна 1 (рядок ~650):
```python
# ЗНАЙТИ:
        from optimized_db import db_manager

# ЗАМІНИТИ НА:
        from optimized.database import DatabaseConnection
        db_manager = DatabaseConnection()
```

---

### tests/test_optimized_db.py - 2 заміни

#### Варіант А: Оновити імпорти
```python
# ЗНАЙТИ (рядок ~19):
    from optimized_db import OptimizedDatabaseManager

# ЗАМІНИТИ НА:
    from optimized.database import DatabaseConnection as OptimizedDatabaseManager

# ЗНАЙТИ (рядок ~67):
        from optimized_db import save_trading_signal

# ЗАМІНИТИ НА:
        from optimized.database import DatabaseConnection
        db = DatabaseConnection()
        save_trading_signal = db.save_signal
```

#### Варіант Б: Перемістити файл
```bash
mkdir -p optimized/database/tests
mv tests/test_optimized_db.py optimized/database/tests/test_database.py
# І оновити імпорти як у Варіанті А
```

---

## 🎯 Результат Після Очищення

### Видалено:
```
❌ resume_training.py              (deprecated, не працює)
❌ optimized_model.py              (wrapper, deprecated)
❌ optimized_db.py                 (wrapper, deprecated)
❌ optimized_indicators.py         (wrapper, deprecated)
❌ tests/debug_scaler.py           (старий debug)
❌ tests/test_scaler_simple.py     (старий test)
```

### Архівовано:
```
📦 archive/ → archive_backup_2025-10-24.tar.gz
```

### Оновлено імпорти:
```
✅ main.py: 5 імпортів оновлено
✅ async_architecture.py: 1 імпорт оновлено
✅ tests/test_optimized_db.py: 2 імпорти оновлено (або перенесено)
```

### Структура після очищення:
```
/home/ihor/data_proj/data/
├── analytics/              ✅ Організовано
├── training/               ✅ Сучасна структура
├── intelligent_sys/        ✅ Торгова система
├── strategies/             ✅ Використовується (17 імпортів)
├── trading/                ✅ Використовується (8 імпортів)
├── optimized/              ✅ Нова структура (БЕЗ wrappers)
├── fast_indicators/        ✅ Rust
├── models/                 ✅ Моделі
├── main.py                 ✅ Оновлені імпорти
├── async_architecture.py   ✅ Оновлені імпорти
└── archive_backup.tar.gz   📦 Архів
```

---

## ⚠️ ВАЖЛИВО

1. **НЕ видаляти optimized_*.py ДО заміни імпортів!**
2. **Створити backup перед діями**
3. **Тестувати після кожної заміни:**
   ```bash
   python main.py --help
   python -c "import main; print('OK')"
   ```
4. **Перевірити що код запускається без помилок**

---

## 📈 Метрики Очищення

| Метрика | До | Після | Різниця |
|---------|-----|-------|---------|
| Python файлів | 112 | 106 | -6 |
| Deprecated wrappers | 3 | 0 | -3 |
| Deprecated скриптів | 1 | 0 | -1 |
| Старих тестів | 2 | 0 | -2 |
| Імпортів оновлено | 0 | 8 | +8 |

**Економія місця:** ~20-50 KB (без archive/)  
**З archive tar.gz:** ~1-5 MB залежно від вмісту

---

**Готово до виконання!** 🚀
