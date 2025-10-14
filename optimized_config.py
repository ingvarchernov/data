# -*- coding: utf-8 -*-
"""
Оптимізована конфігурація - МАКСИМАЛЬНА для GTX 1050 4GB
"""

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS_BACK = 90  # 3 місяці історії для кращого навчання моделі
LOOK_BACK = 168  # 1 тиждень історії - оптимально для 90 днів даних
STEPS = 5

MODEL_CONFIG = {
    "model_type": "advanced_lstm",  # Змінено з transformer_lstm на простішу модель
    "lstm_units_1": 384,  # Трохи зменшено для GPU memory
    "lstm_units_2": 192,  # Трохи зменшено
    "lstm_units_3": 96,  # Трохи зменшено
    "attention_heads": 8,  # Збільшено
    "attention_key_dim": 32,  # Збільшено
    "dense_units": [256, 128, 64],  # Розширено
    "epochs": 10,  # Збільшено для кращого навчання
    "batch_size": 64,  # Зменшено для GTX 1050 (3.4GB VRAM)
    "learning_rate": 0.0003,  # Ще зменшено для кращої стабільності
    "patience": 15,  # Збільшено
    "validation_split": 0.2  # Залишаємо
}

INDICATORS_CONFIG = {
    "rsi_period": 14,
    "ema_period": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_smooth_k": 3,
    "stoch_smooth_d": 3,
    "atr_period": 14,
    "momentum_period": 10,  # Новий індикатор
    "roc_period": 10,       # Rate of Change
    "williams_r_period": 14,
    "cci_period": 20,       # Commodity Channel Index
    "trix_period": 15,      # Triple Exponential Average
    "keltner_period": 20,   # Keltner Channels
    "keltner_atr": 2.0
}

CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "default_ttl": 3600,
    "memory_cache_size": 1000,
    "use_redis": True,
    "use_memory": True
}

DB_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600
}

ASYNC_CONFIG = {
    "max_workers": 6,
    "thread_pool_size": 12,
    "process_pool_size": 4,
    "task_queue_size": 1000
}

GPU_CONFIG = {
    "use_mixed_precision": True,
    "use_xla": False,  # Вимкнено через memory issues
    "memory_growth": True
}
