# -*- coding: utf-8 -*-
"""
Оптимізована конфігурація системи
"""

# Базові параметри торгівлі
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS_BACK = 365
LOOK_BACK = 360
STEPS = 5

# Параметри ML моделі
MODEL_CONFIG = {
    "model_type": "transformer_lstm",  # "transformer_lstm", "advanced_lstm", "cnn_lstm"
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "patience": 50,
    "validation_split": 0.2
}

# Параметри технічних індикаторів
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
    "atr_period": 14
}

# Параметри кешування
CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "default_ttl": 3600,  # секунди
    "memory_cache_size": 1000,
    "use_redis": True,
    "use_memory": True
}

# Параметри бази даних
DB_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600
}

# Параметри асинхронної системи
ASYNC_CONFIG = {
    "max_workers": 6,
    "thread_pool_size": 12,
    "process_pool_size": 4,
    "task_queue_size": 1000
}

# Параметри GPU
GPU_CONFIG = {
    "use_mixed_precision": True,
    "use_xla": True,
    "memory_growth": True
}

# Параметри системних ресурсів
RESOURCE_CONFIG = {
    "cpu_threshold": 80.0,  # %
    "memory_threshold": 85.0,  # %
    "gpu_memory_threshold": 90.0,  # %
}

# Параметри логування
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "optimized_app.log",
    "max_size": "100MB",
    "backup_count": 5
}

# Шляхи файлів
PATHS = {
    "models": "models/",
    "logs": "logs/",
    "data": "data/",
    "cache": "cache/",
    "exports": "exports/"
}

# Параметри безпеки
SECURITY_CONFIG = {
    "max_retries": 3,
    "timeout": 30,
    "rate_limit": 100,  # запитів на хвилину
    "encrypt_cache": False
}

# Параметри моніторингу
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_interval": 60,  # секунди
    "alert_thresholds": {
        "error_rate": 5.0,  # %
        "response_time": 10.0,  # секунди
        "memory_usage": 90.0,  # %
        "cpu_usage": 85.0  # %
    }
}