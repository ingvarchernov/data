#!/usr/bin/env python3
"""
Advanced trainer з Rust індикаторами
Використовує BaseModelTrainer + FeatureEngineer + Rust (якщо доступний)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# TensorFlow imports - compatible with both TF 2.x versions
try:
    import tensorflow as tf
    from tensorflow import keras
    layers = keras.layers
    Model = keras.Model
except ImportError:
    raise ImportError("TensorFlow is required. Install: pip install tensorflow")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training import BaseModelTrainer, FeatureEngineer
from gpu_config import configure_gpu

logger = logging.getLogger(__name__)
configure_gpu()

# Конфігурація з advanced_training_config.py
ADVANCED_CONFIG = {
    'model_type': 'deep_lstm_attention',
    'sequence_length': 96,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 0.0003,
    'early_stopping_patience': 30,
    'reduce_lr_patience': 12,
    
    # Deep LSTM
    'lstm_units': [256, 192, 128, 96],
    
    # Attention
    'attention_heads': 12,
    'attention_key_dim': 96,
    
    # Dense
    'dense_units': [512, 256, 128, 64],
    
    # Regularization
    'dropout_rate': 0.45,
    'l2_regularization': 0.012,
}


class AdvancedTrainer(BaseModelTrainer):
    """
    Advanced trainer з розширеними можливостями:
    - Rust індикатори для швидкодії (опціонально)
    - Deep LSTM architecture
    - Multi-head attention
    - Advanced feature engineering
    """
    
    def __init__(self, symbol: str, interval: str = '1h', config: Dict = None, use_rust: bool = True):
        super().__init__(symbol, interval)
        self.config = ADVANCED_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.feature_engineer = FeatureEngineer()
        self.use_rust = use_rust
        self.rust_available = False
        
        # Перевірка Rust модуля
        if use_rust:
            try:
                import fast_indicators
                self.rust_available = True
                self.fast_indicators = fast_indicators
                logger.info("✅ Використовуємо Rust індикатори для швидкодії")
            except ImportError:
                logger.warning("⚠️ Rust модуль недоступний, використовується Python")
        
        logger.info(f"📊 AdvancedTrainer: {symbol}, "
                   f"seq_len={self.config['sequence_length']}, "
                   f"batch={self.config['batch_size']}, "
                   f"rust={'✅' if self.rust_available else '❌'}")
    
    def add_rust_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає розширені індикатори через Rust"""
        if not self.rust_available:
            logger.debug("Rust недоступний, пропускаємо розширені індикатори")
            return df
        
        try:
            prices = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Helper для додавання індикаторів
            def add_indicator(name: str, values: list):
                if len(values) > 0:
                    if len(values) < len(df):
                        padding = [np.nan] * (len(df) - len(values))
                        df[name] = padding + list(values)
                    else:
                        df[name] = values
            
            # RSI з додатковими періодами
            add_indicator('rust_rsi_7', self.fast_indicators.fast_rsi(prices, 7))
            add_indicator('rust_rsi_21', self.fast_indicators.fast_rsi(prices, 21))
            
            # MACD компоненти
            macd, signal, hist = self.fast_indicators.fast_macd(prices, 12, 26, 9)
            add_indicator('rust_macd', macd)
            add_indicator('rust_macd_signal', signal)
            add_indicator('rust_macd_histogram', hist)
            
            # Bollinger Bands + width & percent
            upper, lower = self.fast_indicators.fast_bollinger_bands(prices, 20, 2.0)
            add_indicator('rust_bb_upper', upper)
            add_indicator('rust_bb_lower', lower)
            if len(upper) > 0 and len(lower) > 0:
                df['rust_bb_width'] = upper - lower
                df['rust_bb_percent'] = (prices[-len(upper):] - lower) / (upper - lower + 1e-8)
            
            # Stochastic
            try:
                stoch_k, stoch_d = self.fast_indicators.fast_stochastic(high, low, prices, 14, 3, 3)
                add_indicator('rust_stoch_k', stoch_k)
                add_indicator('rust_stoch_d', stoch_d)
            except Exception as e:
                logger.debug(f"Stochastic пропущено: {e}")
            
            logger.info(f"✅ Додано {len([c for c in df.columns if c.startswith('rust_')])} Rust індикаторів")
            
        except Exception as e:
            logger.warning(f"⚠️ Помилка Rust індикаторів: {e}")
        
        return df
    
    async def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розширена підготовка фічей:
        1. Python індикатори через FeatureEngineer
        2. Rust індикатори (якщо доступні)
        3. Advanced price patterns
        """
        logger.info("🔧 Розрахунок розширених індикаторів...")
        
        # 1. Базові індикатори через FeatureEngineer
        df = self.feature_engineer.calculate_all(df)
        
        # 2. Rust індикатори (опціонально)
        if self.rust_available:
            df = self.add_rust_indicators(df)
        
        # 3. Advanced price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_change'] = df['volume'].pct_change()
        
        # Price momentum features
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
        
        # Volatility features
        for period in [7, 14, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volume_volatility_{period}'] = df['volume_change'].rolling(period).std()
        
        # Drop rows with NaN
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"✅ Features: {len(df.columns)}, rows: {initial_len} → {len(df)}")
        
        return df
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Deep LSTM з Multi-Head Attention
        
        Returns:
            keras.Model: Compiled model
        
        Architecture:
        - 4x Bidirectional LSTM layers (deep architecture)
        - Multi-head Attention механізм
        - Dense layers з residual connections
        - Heavy regularization
        """
        seq_len, n_features = input_shape
        config = self.config
        
        logger.info(f"🏗️ Building Deep LSTM + Attention: input_shape={input_shape}")
        
        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs
        
        # Deep LSTM blocks
        for i, units in enumerate(config['lstm_units']):
            return_sequences = (i < len(config['lstm_units']) - 1) or True  # Attention потребує sequences
            
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization'])
                ),
                name=f'bi_lstm_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
        
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
        
        # Dense blocks з residual connections
        for i, units in enumerate(config['dense_units']):
            residual = x
            
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization']),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout_rate'])(x)
            x = layers.BatchNormalization()(x)
            
            # Residual connection (якщо розмірності співпадають)
            if residual.shape[-1] == units:
                x = layers.Add()([x, residual])
        
        # Output
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AdvancedDeepLSTM')
        
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
    
    async def train(self, days: int = 730) -> Dict:
        """Тренування з extended dataset (2 роки)"""
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
        
        logger.info(f"📊 Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logger.info(f"🎯 Target: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        logger.info(f"📈 Features: {len(feature_names)}")
        
        # 6. Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # 7. Callbacks
        callbacks = self.get_callbacks(
            model_dir=f'models/advanced_{self.symbol}',
            early_stopping_patience=self.config['early_stopping_patience'],
            reduce_lr_patience=self.config['reduce_lr_patience']
        )
        
        # 8. Train
        logger.info("🚀 Starting advanced training...")
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
            model_dir=f'models/advanced_{self.symbol}',
            scaler=self.scaler,
            feature_names=feature_names
        )
        
        return {
            'history': history.history,
            'test_results': test_results,
            'config': self.config,
            'feature_names': feature_names,
            'rust_used': self.rust_available
        }


async def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train advanced model with Rust indicators')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--interval', type=str, default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    parser.add_argument('--no-rust', action='store_true', help='Disable Rust indicators')
    args = parser.parse_args()
    
    trainer = AdvancedTrainer(
        symbol=args.symbol,
        interval=args.interval,
        use_rust=not args.no_rust
    )
    
    results = await trainer.train(days=args.days)
    
    logger.info("\n" + "="*60)
    logger.info("📊 ADVANCED TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Features: {len(results['feature_names'])}")
    logger.info(f"Rust indicators: {'✅' if results['rust_used'] else '❌'}")
    logger.info(f"Test Loss: {results['test_results']['loss']:.4f}")
    logger.info(f"Test MAE: {results['test_results']['mae']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
