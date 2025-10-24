#!/usr/bin/env python3
"""
Classification Trainer з ПРАВИЛЬНИМИ лейблами
Виправляє проблему інверсії сигналів
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime

# TensorFlow imports - compatible with both TF 2.x versions
try:
    import tensorflow as tf
    from tensorflow import keras
    layers = keras.layers
    Model = keras.Model
except ImportError:
    raise ImportError("TensorFlow is required. Install: pip install tensorflow")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.base_trainer import BaseModelTrainer
from training.feature_engineering import FeatureEngineer
from training.rust_features import RustFeatureEngineer
from training.utils import create_classification_targets, calculate_class_weights
from gpu_config import configure_gpu

logger = logging.getLogger(__name__)
configure_gpu()

# Конфігурація
CLASSIFICATION_CONFIG = {
    'model_type': 'classification_lstm_attention',
    'sequence_length': 120,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 0.0003,
    'early_stopping_patience': 25,
    'reduce_lr_patience': 10,
    
    # LSTM
    'lstm_units': [128, 96, 64],
    
    # Attention
    'attention_heads': 8,
    'attention_key_dim': 64,
    
    # Dense
    'dense_units': [256, 128, 64],
    
    # Regularization
    'dropout_rate': 0.4,
    'l2_regularization': 0.01,
    
    # Classification
    'num_classes': 3,  # DOWN, NEUTRAL, UP
    'down_threshold': -0.007,  # -0.7% (компроміс між 0.5% і 1%)
    'up_threshold': 0.007,     # +0.7% (компроміс між 0.5% і 1%)
    
    # Class weights - більш агресивні
    'use_class_weights': True,
}


class ClassificationTrainer(BaseModelTrainer):
    """
    Класифікаційний тренер з правильними лейблами
    
    ВАЖЛИВО: Лейбли створюються ПРАВИЛЬНО:
    - UP = якщо майбутня ціна ЗРОСТЕ
    - DOWN = якщо майбутня ціна ВПАДЕ
    - NEUTRAL = якщо зміна в межах порогів
    """
    
    def __init__(self, symbol: str, interval: str = '1h', config: Dict = None, use_rust: bool = True):
        super().__init__(symbol, interval)
        
        self.config = {**CLASSIFICATION_CONFIG, **(config or {})}
        self.use_rust = use_rust
        self.rust_available = False
        
        # Feature engineers
        self.feature_engineer = FeatureEngineer()
        
        if use_rust:
            try:
                self.rust_engineer = RustFeatureEngineer()
                self.rust_available = True
                logger.info("🦀 Rust індикатори доступні")
            except Exception as e:
                logger.warning(f"⚠️ Rust індикатори недоступні: {e}")
                self.rust_available = False
    
    async def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок всіх features"""
        logger.info("🔧 Розрахунок features...")
        
        # 1. Базові індикатори
        if self.rust_available:
            logger.info("   🦀 Використання Rust індикаторів...")
            df = self.rust_engineer.calculate_all(df)
        else:
            logger.info("   🐍 Використання Python індикаторів...")
            df = self.feature_engineer.calculate_all(df)
        
        # 2. Додаткові features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_change'] = df['volume'].pct_change()
        
        # Momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Volatility
        for period in [7, 14, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Drop NaN
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"✅ Features: {len(df.columns)}, rows: {initial_len} → {len(df)}")
        
        return df
    
    def create_classification_target(self, df: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Створення ПРАВИЛЬНИХ класифікаційних таргетів
        
        ВАЖЛИВО: future_return > 0 → UP (ціна ЗРОСТЕ)
                 future_return < 0 → DOWN (ціна ВПАДЕ)
        """
        logger.info(f"🎯 Створення classification targets (horizon={prediction_horizon})...")
        
        # Розрахунок майбутнього return
        df['future_price'] = df['close'].shift(-prediction_horizon)
        df['future_return'] = (df['future_price'] / df['close']) - 1
        
        # Створення класів
        down_threshold = self.config['down_threshold']
        up_threshold = self.config['up_threshold']
        
        conditions = [
            df['future_return'] < down_threshold,      # DOWN (0)
            (df['future_return'] >= down_threshold) & (df['future_return'] <= up_threshold),  # NEUTRAL (1)
            df['future_return'] > up_threshold         # UP (2)
        ]
        
        df['target_class'] = np.select(conditions, [0, 1, 2], default=1)
        
        # Статистика
        class_counts = df['target_class'].value_counts().sort_index()
        total = len(df[~df['target_class'].isna()])
        
        logger.info(f"📊 Розподіл класів:")
        logger.info(f"   DOWN (0):    {class_counts.get(0, 0):5d} ({class_counts.get(0, 0)/total*100:5.1f}%)")
        logger.info(f"   NEUTRAL (1): {class_counts.get(1, 0):5d} ({class_counts.get(1, 0)/total*100:5.1f}%)")
        logger.info(f"   UP (2):      {class_counts.get(2, 0):5d} ({class_counts.get(2, 0)/total*100:5.1f}%)")
        
        # Перевірка на інверсію (для безпеки)
        sample_up = df[df['target_class'] == 2]['future_return'].head(10)
        sample_down = df[df['target_class'] == 0]['future_return'].head(10)
        
        logger.info(f"🔍 Перевірка лейблів:")
        logger.info(f"   UP класс має позитивні returns: {(sample_up > 0).all()}")
        logger.info(f"   DOWN класс має негативні returns: {(sample_down < 0).all()}")
        
        # Видаляємо допоміжні колонки та NaN
        df = df.drop(['future_price'], axis=1, errors='ignore')
        df = df.dropna(subset=['target_class'])
        
        return df
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Класифікаційна модель: Bidirectional LSTM + Multi-Head Attention
        
        Returns:
            keras.Model: Compiled classification model
        """
        seq_len, n_features = input_shape
        config = self.config
        
        logger.info(f"🏗️ Building Classification Model: input={input_shape}, classes={config['num_classes']}")
        
        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs
        
        # LSTM blocks
        for i, units in enumerate(config['lstm_units']):
            return_sequences = (i < len(config['lstm_units']) - 1) or True  # Завжди True для attention
            
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=config['dropout_rate'],
                    recurrent_dropout=0.2,
                    kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
                ),
                name=f'bidirectional_lstm_{i}'
            )(x)
            
            if i < len(config['lstm_units']) - 1:
                x = layers.Dropout(config['dropout_rate'])(x)
        
        # Multi-Head Attention
        attention = layers.MultiHeadAttention(
            num_heads=config['attention_heads'],
            key_dim=config['attention_key_dim'],
            dropout=config['dropout_rate'],
            name='multi_head_attention'
        )(x, x)
        
        # Residual connection
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for i, units in enumerate(config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization']),
                name=f'dense_{i}'
            )(x)
            x = layers.Dropout(config['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
        # Output layer (3 класи: DOWN, NEUTRAL, UP)
        outputs = layers.Dense(
            config['num_classes'],
            activation='softmax',
            name='output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='classification_model')
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        logger.info(f"✅ Model: {model.count_params():,} parameters")
        
        return model
    
    async def train(self, days: int = 365) -> Dict:
        """Тренування класифікаційної моделі"""
        logger.info(f"🚀 Початок тренування класифікаційної моделі для {self.symbol}")
        logger.info(f"📅 Період даних: {days} днів")
        
        # 1. Завантаження даних
        from unified_binance_loader import UnifiedBinanceLoader
        loader = UnifiedBinanceLoader(testnet=True)
        
        df = await loader.get_historical_data(
            symbol=f"{self.symbol}USDT",
            interval=self.interval,
            days_back=days
        )
        
        if df is None or len(df) < 1000:
            raise ValueError(f"Недостатньо даних: {len(df) if df is not None else 0}")
        
        logger.info(f"✅ Завантажено {len(df)} записів")
        
        # 2. Features
        df = await self.prepare_features(df)
        
        # 3. Створення ПРАВИЛЬНИХ таргетів
        df = self.create_classification_target(df, prediction_horizon=1)
        
        # 4. Підготовка даних для послідовностей
        feature_cols = [col for col in df.columns if col not in ['target_class', 'future_return', 
                                                                   'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df[feature_cols].values
        y = df['target_class'].values
        
        logger.info(f"📊 X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"🎯 Features: {len(feature_cols)}")
        
        # 5. Нормалізація
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 6. Створення послідовностей
        sequence_length = self.config['sequence_length']
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"📐 Sequences: {X_sequences.shape}, targets: {y_sequences.shape}")
        
        # 7. Temporal split (80/10/10)
        train_size = int(len(X_sequences) * 0.8)
        val_size = int(len(X_sequences) * 0.1)
        
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        
        X_val = X_sequences[train_size:train_size + val_size]
        y_val = y_sequences[train_size:train_size + val_size]
        
        X_test = X_sequences[train_size + val_size:]
        y_test = y_sequences[train_size + val_size:]
        
        logger.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 8. Class weights
        class_weights = None
        if self.config['use_class_weights']:
            class_weights = calculate_class_weights(y_train)
            logger.info(f"⚖️ Class weights: {class_weights}")
        
        # 9. Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # 10. Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f'models/classification_{self.symbol}'
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{model_dir}/model_checkpoint_{timestamp}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=f'{model_dir}/training_{timestamp}.csv'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/tensorboard_classification_{timestamp}'
            )
        ]
        
        # 11. Training
        logger.info("🎓 Початок навчання...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # 12. Evaluation
        logger.info("📊 Оцінка на test set...")
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"✅ Test Loss: {test_results[0]:.4f}")
        logger.info(f"✅ Test Accuracy: {test_results[1]:.4f}")
        
        # 13. Збереження
        model_path = f'{model_dir}/model_{timestamp}.keras'
        self.model.save(model_path)
        logger.info(f"💾 Модель збережено: {model_path}")
        
        # Збереження scaler та feature names
        import joblib
        joblib.dump(scaler, f'{model_dir}/scaler_{timestamp}.pkl')
        joblib.dump(feature_cols, f'{model_dir}/features_{timestamp}.pkl')
        logger.info(f"💾 Scaler та features збережено")
        
        return {
            'model_path': model_path,
            'test_loss': test_results[0],
            'test_accuracy': test_results[1],
            'history': history.history,
            'timestamp': timestamp
        }


if __name__ == '__main__':
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        trainer = ClassificationTrainer(symbol='BTC', interval='1h')
        results = await trainer.train(days=365)
        print(f"\n✅ Тренування завершено: {results}")
    
    asyncio.run(main())
