"""
Base Model Trainer - Базовий клас для всіх trainers

Надає спільний функціонал:
- Завантаження даних (load_data)
- Підготовка features (prepare_features)
- Створення sequences (create_sequences)
- Train/test split
- Callbacks та логування
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from .data_loader import DataLoader
from .utils import create_sequences, normalize_data, temporal_split, calculate_class_weights

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Базовий клас для тренування моделей
    
    Підкласи мають реалізувати:
    - prepare_features(): специфічна обробка features
    - build_model(): архітектура моделі
    - train(): процес тренування
    """
    
    def __init__(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1h',
        sequence_length: int = 60,
        config: Optional[Dict] = None
    ):
        """
        Args:
            symbol: Торговий символ
            interval: Таймфрейм
            sequence_length: Довжина послідовності
            config: Додаткова конфігурація
        """
        self.symbol = symbol
        self.interval = interval
        self.sequence_length = sequence_length
        self.config = config or {}
        
        # Компоненти
        self.data_loader = None
        self.scaler = None
        self.model = None
        
        # Дані
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        logger.info(f"✅ {self.__class__.__name__} ініціалізовано: {symbol} {interval}")
    
    async def load_data(self, days: int = 365, force_reload: bool = False) -> pd.DataFrame:
        """
        Завантаження даних з Binance
        
        Args:
            days: Кількість днів історії
            force_reload: Примусове завантаження
        
        Returns:
            DataFrame з OHLCV даними
        """
        logger.info(f"📊 Завантаження даних: {self.symbol} {self.interval}, {days} днів")
        
        if self.data_loader is None:
            self.data_loader = DataLoader(
                cache_dir=self.config.get('cache_dir'),
                use_cache=self.config.get('use_cache', False)
            )
        
        df = await self.data_loader.load(
            symbol=self.symbol,
            interval=self.interval,
            days=days,
            force_reload=force_reload
        )
        
        logger.info(f"✅ Завантажено {len(df)} записів")
        return df
    
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Підготовка features (має бути реалізовано в підкласах)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame з features
        """
        pass
    
    def create_target(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        prediction_horizon: int = 1
    ) -> pd.Series:
        """
        Створення target variable
        
        Args:
            df: DataFrame з даними
            target_column: Колонка для прогнозу
            prediction_horizon: Горизонт прогнозу (кількість періодів вперед)
        
        Returns:
            Series з target values
        """
        # За замовчуванням - returns
        target = df[target_column].shift(-prediction_horizon) / df[target_column] - 1
        return target
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення sequences та нормалізація
        
        Args:
            X: Features array
            y: Target array
            normalize: Чи нормалізувати features
        
        Returns:
            X_sequences, y_sequences
        """
        # Нормалізація
        if normalize:
            X, self.scaler = normalize_data(
                X,
                scaler_type=self.config.get('scaler_type', 'robust'),
                fit=True
            )
        
        # Створення sequences
        X_seq, y_seq = create_sequences(
            X, y,
            sequence_length=self.sequence_length,
            step=self.config.get('sequence_step', 1)
        )
        
        logger.info(f"✅ Створено sequences: X{X_seq.shape}, y{y_seq.shape}")
        return X_seq, y_seq
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Розділення на train/val/test sets
        
        Args:
            X: Features
            y: Targets
            train_ratio: Розмір train set
            val_ratio: Розмір validation set
        """
        # Розрахунок індексів
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Temporal split (без shuffle для time-series)
        self.X_train = X[:train_end]
        self.X_val = X[train_end:val_end]
        self.X_test = X[val_end:]
        
        self.y_train = y[:train_end]
        self.y_val = y[train_end:val_end]
        self.y_test = y[val_end:]
        
        logger.info(f"✅ Дані розділено:")
        logger.info(f"   Train: {len(self.X_train)} samples")
        logger.info(f"   Val:   {len(self.X_val)} samples")
        logger.info(f"   Test:  {len(self.X_test)} samples")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Побудова архітектури моделі (має бути реалізовано в підкласах)
        
        Returns:
            Compiled Keras model
        """
        pass
    
    def get_callbacks(self, model_dir: Path) -> List[tf.keras.callbacks.Callback]:
        """
        Створення callbacks для тренування
        
        Args:
            model_dir: Директорія для збереження моделі
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = model_dir / f'model_{self.symbol}_{self.interval}.keras'
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        if self.config.get('early_stopping', True):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping_patience', 20),
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Reduce LR on plateau
        if self.config.get('reduce_lr', True):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.get('reduce_lr_patience', 10),
                    min_lr=1e-7,
                    verbose=1
                )
            )
        
        # CSV Logger
        csv_path = model_dir / f'training_{self.symbol}_{self.interval}.csv'
        callbacks.append(
            tf.keras.callbacks.CSVLogger(csv_path)
        )
        
        return callbacks
    
    @abstractmethod
    async def train(self, **kwargs):
        """
        Процес тренування (має бути реалізовано в підкласах)
        """
        pass
    
    def evaluate(self) -> Dict:
        """
        Оцінка моделі на test set
        
        Returns:
            Dict з метриками
        """
        if self.model is None:
            raise ValueError("Модель не натренована")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test set не підготовлено")
        
        logger.info("\n📊 Оцінка на test set...")
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        metrics = {}
        if isinstance(results, list):
            metrics['test_loss'] = results[0]
            for i, metric_name in enumerate(self.model.metrics_names[1:], 1):
                metrics[f'test_{metric_name}'] = results[i]
        else:
            metrics['test_loss'] = results
        
        for key, value in metrics.items():
            logger.info(f"   {key}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path: str):
        """Збереження моделі"""
        if self.model is None:
            raise ValueError("Модель не існує")
        
        self.model.save(path)
        logger.info(f"💾 Модель збережено: {path}")
    
    def load_model(self, path: str):
        """Завантаження моделі"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"✅ Модель завантажено: {path}")
    
    async def cleanup(self):
        """Очистка ресурсів"""
        if self.data_loader:
            await self.data_loader.close()
        logger.info("🔒 Cleanup завершено")
