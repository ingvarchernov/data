#!/usr/bin/env python3
"""
Оптимізований trainer з відібраними фічами
Використовує BaseModelTrainer + FeatureEngineer
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training import BaseModelTrainer, FeatureEngineer
from selected_features import SELECTED_FEATURES
from gpu_config import configure_gpu

logger = logging.getLogger(__name__)
configure_gpu()

# ОПТИМАЛЬНА КОНФІГУРАЦІЯ
OPTIMAL_CONFIG = {
    'model_type': 'advanced_lstm',
    'sequence_length': 60,
    'batch_size': 64,
    'epochs': 200,
    'learning_rate': 0.0005,
    'early_stopping_patience': 25,
    'reduce_lr_patience': 10,
    
    # LSTM
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


class OptimizedTrainer(BaseModelTrainer):
    """Trainer з відібраними топ-35 фічами"""
    
    def __init__(self, symbol: str, interval: str = '1h', config: Dict = None):
        super().__init__(symbol, interval)
        self.config = OPTIMAL_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.feature_engineer = FeatureEngineer()
        
        logger.info(f"📊 OptimizedTrainer: {symbol}, "
                   f"seq_len={self.config['sequence_length']}, "
                   f"batch={self.config['batch_size']}, "
                   f"features={len(SELECTED_FEATURES)}")
    
    async def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Підготовка features з топ-35 features
        
        Args:
            df: DataFrame з OHLCV даними
            
        Returns:
            DataFrame з розрахованими індикаторами
        """
        # Розрахунок всіх індикаторів з правильними періодами
        df = self.feature_engineer.calculate_all(
            df,
            sma_periods=[10, 20, 50, 200],  # Include sma_10
            ema_periods=[12, 20, 26, 50],
            rsi_periods=[7, 14, 28],
            atr_periods=[7, 14, 28]
        )
        
        # Базові price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_change'] = df['volume'].pct_change()
        
        # Відбір тільки SELECTED_FEATURES
        available_features = [f for f in SELECTED_FEATURES if f in df.columns]
        missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Відсутні фічі: {missing_features}")
        
        logger.info(f"✅ Використовується {len(available_features)} фічей з {len(SELECTED_FEATURES)}")
        
        # Залишаємо тільки відібрані фічі + ціна для target
        df = df[['close'] + available_features].copy()
        df = df.dropna()
        
        return df
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Побудова Advanced LSTM моделі з Attention
        
        Architecture:
        - 3x Bidirectional LSTM layers with decreasing units
        - Multi-head Attention
        - Dense layers with BatchNorm and Dropout
        - L2 regularization
        """
        seq_len, n_features = input_shape
        config = self.config
        
        logger.info(f"🏗️ Building Advanced LSTM model: input_shape={input_shape}")
        
        # Input
        inputs = layers.Input(shape=input_shape, name='input')
        
        # LSTM Block 1
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units_1'], 
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
            ),
            name='bi_lstm_1'
        )(inputs)
        x = layers.Dropout(config['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        # LSTM Block 2
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units_2'],
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
            ),
            name='bi_lstm_2'
        )(x)
        x = layers.Dropout(config['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        
        # LSTM Block 3
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units_3'],
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
            ),
            name='bi_lstm_3'
        )(x)
        x = layers.Dropout(config['dropout_rate'])(x)
        
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=config['attention_heads'],
            key_dim=config['attention_key_dim'],
            name='multi_head_attention'
        )(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for i, units in enumerate(config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization']),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
        # Output
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # Compile
        model = Model(inputs=inputs, outputs=outputs, name='OptimizedAdvancedLSTM')
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"✅ Model built: {model.count_params():,} parameters")
        
        return model
    
    async def train(self, days: int = 365) -> Dict:
        """
        Тренування моделі
        
        Returns:
            dict: Training results with history and metrics
        """
        # 1. Load data
        df = await self.load_data(days=days)
        if df is None:
            raise ValueError("Failed to load data")
        
        # 2. Prepare features
        df = await self.prepare_features(df)
        
        # 3. Create target
        df = self.create_target(df, shift_periods=1)
        
        # 4. Prepare sequences
        X, y, feature_names = self.prepare_sequences(
            df, 
            sequence_length=self.config['sequence_length']
        )
        
        # 5. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        logger.info(f"📊 Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logger.info(f"🎯 Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        
        # 6. Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # 7. Callbacks
        callbacks = self.get_callbacks(
            model_dir=f'models/optimized_{self.symbol}',
            early_stopping_patience=self.config['early_stopping_patience'],
            reduce_lr_patience=self.config['reduce_lr_patience']
        )
        
        # 8. Train
        logger.info("🚀 Starting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 9. Evaluate
        test_results = self.evaluate(X_test, y_test)
        
        # 10. Save
        self.save_model(
            model_dir=f'models/optimized_{self.symbol}',
            scaler=self.scaler,
            feature_names=feature_names
        )
        
        return {
            'history': history.history,
            'test_results': test_results,
            'config': self.config,
            'feature_names': feature_names
        }


async def main():
    """Головна функція для тренування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train optimized model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--interval', type=str, default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    args = parser.parse_args()
    
    # Train
    trainer = OptimizedTrainer(symbol=args.symbol, interval=args.interval)
    results = await trainer.train(days=args.days)
    
    logger.info("\n" + "="*60)
    logger.info("📊 TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Features: {len(results['feature_names'])}")
    logger.info(f"Test Loss: {results['test_results']['loss']:.4f}")
    logger.info(f"Test MAE: {results['test_results']['mae']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
