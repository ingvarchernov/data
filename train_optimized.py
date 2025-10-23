#!/usr/bin/env python3
"""
Оптимізоване тренування з відібраними фічами
Використовує топ-35 фічей + робочу конфігурацію
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_config import configure_gpu
from optimized_model import OptimizedPricePredictionModel
from train_model_advanced import AdvancedModelTrainer
from selected_features import SELECTED_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

configure_gpu()

# ОПТИМАЛЬНА КОНФІГУРАЦІЯ (повернення до робочої)
OPTIMAL_CONFIG = {
    'model_type': 'advanced_lstm',  # НЕ deep_lstm_xl!
    'sequence_length': 60,           # НЕ 96!
    'batch_size': 64,                # НЕ 32!
    'epochs': 200,
    'learning_rate': 0.0005,
    'early_stopping_patience': 25,
    'reduce_lr_patience': 10,
    
    # LSTM (робоча конфігурація)
    'lstm_units_1': 320,
    'lstm_units_2': 160,
    'lstm_units_3': 80,
    
    # Attention
    'attention_heads': 10,
    'attention_key_dim': 80,
    
    # Dense
    'dense_units': [640, 320, 160, 80],
    
    # Regularization
    'dropout_rate': 0.4,
    'l2_regularization': 0.01,
}


class OptimizedModelTrainer:
    """Оптимізований trainer з відібраними фічами"""
    
    def __init__(self, symbol: str, interval: str = '1h'):
        self.symbol = symbol
        self.interval = interval
        self.config = OPTIMAL_CONFIG.copy()
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        
        logger.info(f"📊 {symbol}: {self.config['model_type']}, "
                   f"seq_len={self.config['sequence_length']}, "
                   f"batch={self.config['batch_size']}, "
                   f"features={len(SELECTED_FEATURES)}")
    
    async def load_data(self, days: int = 365) -> pd.DataFrame:
        """Завантаження даних"""
        logger.info(f"📥 Завантаження {days} днів даних для {self.symbol}...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        from intelligent_sys import UnifiedBinanceLoader
        
        loader = UnifiedBinanceLoader(
            api_key=os.getenv('FUTURES_API_KEY'),
            api_secret=os.getenv('FUTURES_API_SECRET'),
            testnet=False
        )
        
        try:
            data = await loader.get_historical_data(
                symbol=self.symbol,
                interval=self.interval,
                days_back=days
            )
            
            if data is None or len(data) < 500:
                logger.error(f"❌ Недостатньо даних для {self.symbol}")
                return None
            
            logger.info(f"✅ Завантажено {len(data)} записів")
            return data
            
        finally:
            await loader.close()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Підготовка ТІЛЬКИ відібраних фічей"""
        logger.info(f"🔧 Розрахунок відібраних фічей для {self.symbol}...")
        
        # Використовуємо існуючий AdvancedModelTrainer для розрахунку всіх фічей
        temp_trainer = AdvancedModelTrainer(self.symbol, testnet=False)
        df_all = temp_trainer.prepare_features(data)
        
        # Відбираємо ТІЛЬКИ топ-35 фічей
        missing_features = [f for f in SELECTED_FEATURES if f not in df_all.columns]
        if missing_features:
            logger.warning(f"⚠️ Відсутні фічі: {missing_features}")
        
        available_features = [f for f in SELECTED_FEATURES if f in df_all.columns]
        df = df_all[available_features].copy()
        
        # Видаляємо NaN
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"✅ Підготовлено {len(df)} записів з {len(df.columns)} відібраними фічами (dropped {dropped} NaN)")
        logger.info(f"📊 Використано фічей: {len(available_features)}/{len(SELECTED_FEATURES)}")
        
        return df
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Створення послідовностей"""
        sequence_length = self.config['sequence_length']
        
        # Всі колонки - це вже відібрані фічі
        feature_cols = list(data.columns)
        
        # Нормалізація
        scaled_data = self.scaler.fit_transform(data[feature_cols].values)
        
        # Створюємо послідовності
        X, y = [], []
        
        # Додаємо колонку close для розрахунку target
        # Знаходимо close в оригінальних даних через index
        close_prices = data.index.to_series().apply(
            lambda idx: data.loc[idx, 'open'] if 'open' in data.columns else 0
        )
        
        # Якщо є returns, використовуємо його для відновлення close
        if 'returns' in data.columns:
            # Відновлюємо close через returns
            logger.info("📊 Використовую returns для target")
        
        for i in range(sequence_length, len(scaled_data) - 1):
            X.append(scaled_data[i-sequence_length:i])
            
            # Target - returns на наступному кроці (вже є в даних)
            if i + 1 < len(data) and 'returns' in data.columns:
                y.append(data['returns'].iloc[i + 1])
            else:
                # Fallback: використовуємо returns поточного кроку
                y.append(data['returns'].iloc[i] if 'returns' in data.columns else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Створено {len(X)} послідовностей, розмір: {X.shape}")
        return X, y
    
    def build_and_compile_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Створення моделі"""
        logger.info(f"🤖 Створення моделі: {self.config['model_type']}")
        
        model_builder = OptimizedPricePredictionModel(
            input_shape=input_shape,
            model_type=self.config['model_type'],
            model_config=self.config
        )
        
        model = model_builder.create_model()
        model = model_builder.compile_model(
            model, 
            learning_rate=self.config['learning_rate']
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Тренування"""
        logger.info(f"🚀 Початок тренування {self.symbol}...")
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        from tensorflow.keras.callbacks import (
            EarlyStopping, 
            ReduceLROnPlateau, 
            ModelCheckpoint
        )
        
        model_dir = f'models/optimized_{self.symbol.replace("USDT", "")}'
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = f'{model_dir}/best_model.h5'
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Останній fold для валідації
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logger.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Побудова моделі
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_and_compile_model(input_shape)
        
        # Тренування
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        # Збереження scaler
        scaler_path = f'{model_dir}/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler збережено: {scaler_path}")
        
        val_loss = min(history.history['val_loss'])
        val_dir_acc = max(history.history.get('val_directional_accuracy', [0]))
        
        logger.info(f"✅ Тренування завершено!")
        logger.info(f"   Best val_loss: {val_loss:.6f}")
        logger.info(f"   Best val_directional_accuracy: {val_dir_acc:.4f}")
        
        return {
            'symbol': self.symbol,
            'val_loss': val_loss,
            'val_dir_acc': val_dir_acc,
            'model_path': checkpoint_path,
            'scaler_path': scaler_path
        }


async def quick_test_optimized():
    """Швидкий тест оптимізованої моделі"""
    symbol = 'BTCUSDT'
    
    logger.info("=" * 80)
    logger.info(f"🚀 ОПТИМІЗОВАНЕ ТРЕНУВАННЯ - {symbol}")
    logger.info("=" * 80)
    logger.info(f"\n📊 Конфігурація:")
    logger.info(f"   Model: {OPTIMAL_CONFIG['model_type']}")
    logger.info(f"   Sequence: {OPTIMAL_CONFIG['sequence_length']}")
    logger.info(f"   Batch: {OPTIMAL_CONFIG['batch_size']}")
    logger.info(f"   Features: {len(SELECTED_FEATURES)} (відібрані)")
    logger.info(f"   Data: 365 days")
    logger.info("")
    
    try:
        trainer = OptimizedModelTrainer(symbol)
        
        # Завантаження даних
        data = await trainer.load_data(days=365)
        if data is None:
            return None
        
        # Підготовка фічей
        df = trainer.prepare_features(data)
        
        # Створення послідовностей
        X, y = trainer.create_sequences(df)
        
        # Тренування
        result = trainer.train(X, y)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ТРЕНУВАННЯ ЗАВЕРШЕНО")
        logger.info("=" * 80)
        logger.info(f"\n📊 РЕЗУЛЬТАТИ:")
        logger.info(f"   Символ: {result['symbol']}")
        logger.info(f"   Best val_directional_accuracy: {result['val_dir_acc']:.4f} ({result['val_dir_acc']*100:.2f}%)")
        logger.info(f"   Best val_loss: {result['val_loss']:.6f}")
        logger.info(f"   Модель збережена: {result['model_path']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ ПОМИЛКА: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    result = asyncio.run(quick_test_optimized())
    
    if result:
        logger.info("\n🎉 УСПІХ!")
        sys.exit(0)
    else:
        logger.error("\n❌ НЕВДАЧА")
        sys.exit(1)
