#!/usr/bin/env python3
"""
Покращена система обробки помилок та логування для продакшну
"""
import os
import sys
import logging
import logging.handlers
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import traceback
import json
import asyncio
from functools import wraps
import time


class TradingLogger:
    """
    Розширена система логування для торгової системи
    """

    def __init__(self, name: str = "trading_system", log_level: str = "INFO",
                 log_dir: str = "logs", max_bytes: int = 10*1024*1024,
                 backup_count: int = 5):
        """
        Ініціалізація логера

        Args:
            name: Ім'я логера
            log_level: Рівень логування
            log_dir: Директорія для логів
            max_bytes: Максимальний розмір файлу
            backup_count: Кількість резервних копій
        """
        self.name = name
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Створення директорії
        os.makedirs(log_dir, exist_ok=True)

        # Налаштування логера
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Видалення існуючих хендлерів
        self.logger.handlers.clear()

        # Форматер
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        # Консольний хендлер
        self._setup_console_handler()

        # Файловий хендлер
        self._setup_file_handler()

        # JSON хендлер для структурованих логів
        self._setup_json_handler()

        # Не передавати логи батькам
        self.logger.propagate = False

        self.logger.info(f"🚀 Trading logger ініціалізований: {name}")

    def _setup_console_handler(self):
        """Налаштування консольного виводу"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self):
        """Налаштування файлового виводу"""
        log_file = os.path.join(self.log_dir, f"{self.name}.log")

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def _setup_json_handler(self):
        """Налаштування JSON логування для аналізу"""
        json_file = os.path.join(self.log_dir, f"{self.name}_structured.log")

        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)

        # JSON форматер
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)

    def get_logger(self) -> logging.Logger:
        """Отримання логера"""
        return self.logger

    def log_trade(self, trade_data: Dict[str, Any]):
        """Логування торгової операції"""
        self.logger.info(f"TRADE: {json.dumps(trade_data)}")

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Логування помилки з контекстом"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        self.logger.error(f"ERROR: {json.dumps(error_data)}")

    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Логування продуктивності"""
        perf_data = {
            "operation": operation,
            "duration_ms": duration * 1000,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")


class JSONFormatter(logging.Formatter):
    """JSON форматер для структурованих логів"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }

        # Додавання exception info якщо є
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ErrorHandler:
    """
    Система обробки помилок для торгової системи
    """

    def __init__(self, logger: Optional[TradingLogger] = None):
        """
        Ініціалізація обробника помилок

        Args:
            logger: Логер для запису помилок
        """
        self.logger = logger or TradingLogger("error_handler")
        self.error_counts = {}
        self.max_retries = 3
        self.retry_delays = [1, 2, 5]  # секунди

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                    retry_func: Optional[Callable] = None) -> bool:
        """
        Обробка помилки

        Args:
            error: Помилка
            context: Контекст помилки
            retry_func: Функція для повтору

        Returns:
            True якщо помилку оброблено успішно
        """
        error_type = type(error).__name__

        # Підрахунок помилок
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Логування помилки
        self.logger.log_error(error, context)

        # Повтор спроби якщо є функція
        if retry_func:
            return self._retry_operation(retry_func, error_type)

        return False

    def _retry_operation(self, func: Callable, error_type: str) -> bool:
        """Повтор операції з експоненціальною затримкою"""
        for attempt in range(self.max_retries):
            try:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                time.sleep(delay)

                self.logger.logger.info(f"🔄 Повтор спроби {attempt + 1}/{self.max_retries} для {error_type}")

                result = func()
                self.logger.logger.info(f"✅ Повтор успішний для {error_type}")
                return True

            except Exception as e:
                self.logger.log_error(e, {"attempt": attempt + 1, "error_type": error_type})

        self.logger.logger.error(f"❌ Всі повтори невдалі для {error_type}")
        return False

    def get_error_stats(self) -> Dict[str, int]:
        """Отримання статистики помилок"""
        return self.error_counts.copy()

    def reset_error_counts(self):
        """Скидання лічильників помилок"""
        self.error_counts.clear()


class TradingException(Exception):
    """Базовий клас для торгових помилок"""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


class APIError(TradingException):
    """Помилка API"""
    pass


class DatabaseError(TradingException):
    """Помилка бази даних"""
    pass


class ValidationError(TradingException):
    """Помилка валідації"""
    pass


class InsufficientFundsError(TradingException):
    """Недостатньо коштів"""
    pass


def error_handler(logger: Optional[TradingLogger] = None):
    """
    Декоратор для обробки помилок

    Args:
        logger: Логер для запису помилок
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler_instance = ErrorHandler(logger)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                error_handler_instance.handle_error(e, context)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler_instance = ErrorHandler(logger)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                error_handler_instance.handle_error(e, context)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def performance_monitor(logger: Optional[TradingLogger] = None):
    """
    Декоратор для моніторингу продуктивності

    Args:
        logger: Логер для запису метрик
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger_instance = logger or TradingLogger("performance")
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                metadata = {
                    "function": func.__name__,
                    "async": True
                }
                logger_instance.log_performance(func.__name__, duration, metadata)

                return result

            except Exception as e:
                duration = time.time() - start_time
                metadata = {
                    "function": func.__name__,
                    "async": True,
                    "error": str(e)
                }
                logger_instance.log_performance(func.__name__, duration, metadata)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger_instance = logger or TradingLogger("performance")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                metadata = {
                    "function": func.__name__,
                    "async": False
                }
                logger_instance.log_performance(func.__name__, duration, metadata)

                return result

            except Exception as e:
                duration = time.time() - start_time
                metadata = {
                    "function": func.__name__,
                    "async": False,
                    "error": str(e)
                }
                logger_instance.log_performance(func.__name__, duration, metadata)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Глобальні екземпляри
default_logger = TradingLogger()
default_error_handler = ErrorHandler(default_logger)

# Зручні функції
def log_trade(trade_data: Dict[str, Any]):
    """Логування торгової операції"""
    default_logger.log_trade(trade_data)

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Логування помилки"""
    default_error_handler.handle_error(error, context)

def get_logger(name: str = "trading_system") -> logging.Logger:
    """Отримання логера"""
    return TradingLogger(name).get_logger()


# Приклад використання
if __name__ == "__main__":
    # Ініціалізація логера
    logger = TradingLogger("example")

    # Логування торгової операції
    trade = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.001,
        "price": 50000,
        "timestamp": datetime.now().isoformat()
    }
    logger.log_trade(trade)

    # Використання декораторів
    @error_handler()
    @performance_monitor()
    def example_function():
        """Приклад функції з обробкою помилок та моніторингом"""
        time.sleep(0.1)
        return "success"

    # Виклик функції
    result = example_function()
    print(f"Результат: {result}")

    # Статистика помилок
    print(f"Статистика помилок: {default_error_handler.get_error_stats()}")