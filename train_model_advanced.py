#!/usr/bin/env python3
"""
Покращений скрипт тренування з розширеними фічами для досягнення 60-70% accuracy
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Локальні імпорти
from gpu_config import configure_gpu
from optimized_model import OptimizedPricePredictionModel
from optimized_indicators import calculate_all_indicators
from intelligent_sys import UnifiedBinanceLoader
from advanced_training_config import (
    DATA_CONFIG, 
    SYMBOL_CONFIGS, 
    MODEL_ARCHITECTURES,
    TRAINING_IMPROVEMENTS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

configure_gpu()

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
INTERVAL = '1h'


class AdvancedFeatureEngineer:
    """Покращений feature engineering - використовує Rust індикатори для швидкодії"""
    
    def __init__(self):
        # Перевіряємо наявність Rust модуля
        try:
            import fast_indicators
            self.rust_available = True
            self.fast_indicators = fast_indicators
            logger.info("✅ Використовуємо Rust індикатори для швидкодії")
        except ImportError:
            self.rust_available = False
            logger.warning("⚠️ Rust модуль недоступний, використовується Python")
    
    def add_rust_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає розширені індикатори через Rust (якщо доступний)"""
        if not self.rust_available:
            logger.warning("Rust недоступний, пропускаємо розширені індикатори")
            return df
        
        try:
            # Використовуємо існуючі Rust індикатори
            prices = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Helper function для додавання з правильною довжиною
            def add_indicator(name, values):
                if len(values) > 0:
                    # Додаємо NaN на початку щоб вирівняти довжину
                    if len(values) < len(df):
                        padding = [np.nan] * (len(df) - len(values))
                        df[name] = padding + list(values)
                    else:
                        df[name] = values
            
            # RSI з різними періодами
            add_indicator('rsi_7', self.fast_indicators.fast_rsi(prices, 7))
            add_indicator('rsi_21', self.fast_indicators.fast_rsi(prices, 21))
            
            # MACD
            macd, signal, hist = self.fast_indicators.fast_macd(prices, 12, 26, 9)
            add_indicator('macd', macd)
            add_indicator('macd_signal', signal)
            add_indicator('macd_histogram', hist)
            
            # Bollinger Bands
            upper, lower = self.fast_indicators.fast_bollinger_bands(prices, 20, 2.0)
            add_indicator('bb_upper', upper)
            add_indicator('bb_lower', lower)
            if len(upper) > 0 and len(lower) > 0:
                add_indicator('bb_width', upper - lower)
                add_indicator('bb_percent', (prices[-len(upper):] - lower) / (upper - lower + 1e-8))
            
            # Stochastic (потрібен smooth_d параметр)
            try:
                stoch_k, stoch_d = self.fast_indicators.fast_stochastic(high, low, prices, 14, 3, 3)
                add_indicator('stoch_k', stoch_k)
                add_indicator('stoch_d', stoch_d)
            except:
                pass  # Stochastic може не працювати
            
            # ATR з різними періодами
            add_indicator('atr_7', self.fast_indicators.fast_atr(high, low, prices, 7))
            add_indicator('atr_21', self.fast_indicators.fast_atr(high, low, prices, 21))
            
            # CCI
            add_indicator('cci', self.fast_indicators.fast_cci(high, low, prices, 20))
            
            # OBV (повертає повну довжину)
            df['obv'] = self.fast_indicators.fast_obv(prices, volume)
            
            # ADX
            add_indicator('adx', self.fast_indicators.fast_adx(high, low, prices, 14))
            
            # VWAP (повертає повну довжину)
            df['vwap'] = self.fast_indicators.fast_vwap(prices, volume, high, low)
            
            # EMA з різними періодами (повертає повну довжину)
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'ema_{period}'] = self.fast_indicators.fast_ema(prices, period)
            
            logger.info(f"✅ Додано Rust індикатори (всього {len(df.columns)} колонок)")
            
        except Exception as e:
            logger.error(f"❌ Помилка при розрахунку Rust індикаторів: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    @staticmethod
    def add_rolling_stats(df: pd.DataFrame, windows=[5, 10, 20, 30]) -> pd.DataFrame:
        """Додає rolling статистики (швидкі pandas операції)"""
        for window in windows:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            
            # Normalized distance from rolling stats
            df[f'dist_from_mean_{window}'] = (df['close'] - df[f'close_mean_{window}']) / (df[f'close_std_{window}'] + 1e-8)
        
        return df
    
    @staticmethod
    def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Додає price action patterns (легкі обчислення)"""
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Breakout detection
        df['breakout_up'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['low'].rolling(20).min().shift(1)).astype(int)
        
        # Body/Wick ratios
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Додає додаткові volatility features"""
        # Historic volatility (різні періоди)
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            df[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(period)
        
        # Volatility ratio
        df['volatility_ratio'] = df['hvol_10'] / (df['hvol_30'] + 1e-8)
        
        return df
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Додає momentum features"""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'mom_{period}'] = df['close'] - df['close'].shift(period)
        
        # Acceleration (change in momentum)
        df['acceleration'] = df['mom_10'] - df['mom_10'].shift(1)
        
        # Velocity (rate of price change)
        df['velocity'] = df['close'].diff() / df['close'].shift(1)
        
        return df


class AdvancedModelTrainer:
    """Покращений trainer з усіма новими фічами"""
    
    def __init__(self, symbol: str, interval: str = '1h', testnet: bool = False):
        self.symbol = symbol
        self.interval = interval
        self.testnet = testnet
        
        # Отримуємо symbol-specific config або default
        symbol_config = SYMBOL_CONFIGS.get(symbol, {})
        arch_name = symbol_config.get('architecture', 'advanced_lstm')
        self.config = MODEL_ARCHITECTURES[arch_name].copy()
        
        # Оновлюємо специфічні параметри символу
        for key in ['sequence_length', 'learning_rate', 'batch_size']:
            if key in symbol_config:
                self.config[key] = symbol_config[key]
        
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        self.feature_engineer = AdvancedFeatureEngineer()
        
        logger.info(f"📊 {symbol}: {arch_name}, seq_len={self.config['sequence_length']}, "
                   f"lr={self.config['learning_rate']}, batch={self.config['batch_size']}")
    
    async def load_data(self, days: int = None) -> pd.DataFrame:
        """Завантаження даних"""
        days = days or DATA_CONFIG['days_back']
        logger.info(f"📥 Завантаження {days} днів даних для {self.symbol}...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        loader = UnifiedBinanceLoader(
            api_key=os.getenv('FUTURES_API_KEY'),
            api_secret=os.getenv('FUTURES_API_SECRET'),
            testnet=self.testnet
        )
        
        try:
            data = await loader.get_historical_data(
                symbol=self.symbol,
                interval=self.interval,
                days_back=days
            )
            
            if data is None or len(data) < DATA_CONFIG['min_data_points']:
                logger.error(f"❌ Недостатньо даних для {self.symbol}: {len(data) if data is not None else 0}")
                return None
            
            logger.info(f"✅ Завантажено {len(data)} записів для {self.symbol}")
            return data
            
        finally:
            await loader.close()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Розширена підготовка фічей з використанням Rust індикаторів"""
        logger.info(f"🔧 Розрахунок розширених індикаторів для {self.symbol}...")
        
        # Базові індикатори (з optimized_indicators.py - використовує Rust якщо доступний)
        df = calculate_all_indicators(data)
        
        # Додаткові базові фічі
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        df['volume_momentum'] = df['volume'] - df['volume'].shift(5)
        
        # RUST індикатори (якщо доступний) - найшвидші обчислення
        df = self.feature_engineer.add_rust_indicators(df)
        
        # Додаткові швидкі фічі (pandas)
        df = self.feature_engineer.add_rolling_stats(df)
        df = self.feature_engineer.add_price_patterns(df)
        df = self.feature_engineer.add_volatility_features(df)
        df = self.feature_engineer.add_momentum_features(df)
        
        # Видаляємо NaN
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"✅ Підготовлено {len(df)} записів з {len(df.columns)} фічами (dropped {dropped} NaN)")
        return df
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Створення послідовностей"""
        sequence_length = self.config['sequence_length']
        
        feature_cols = [col for col in data.columns 
                       if col not in ['timestamp', 'open_time', 'close_time']]
        
        scaled_data = self.scaler.fit_transform(data[feature_cols].values)
        
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data) - 1):
            X.append(scaled_data[i-sequence_length:i])
            current_price = data.iloc[i]['close']
            next_price = data.iloc[i + 1]['close']
            price_change = (next_price - current_price) / current_price
            y.append(price_change)
        
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
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Тренування з покращеннями"""
        logger.info(f"🚀 Початок тренування {self.symbol}...")
        
        # Cross-validation
        n_splits = TRAINING_IMPROVEMENTS['validation']['n_splits']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        from tensorflow.keras.callbacks import (
            EarlyStopping, 
            ReduceLROnPlateau, 
            ModelCheckpoint
        )
        
        model_dir = f'models/advanced_{self.symbol.replace("USDT", "")}'
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
        
        # Беремо останній fold
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


async def train_all_models_advanced():
    """Тренування всіх моделей з покращеннями"""
    logger.info("=" * 80)
    logger.info("🚀 ПОКРАЩЕНЕ ТРЕНУВАННЯ МОДЕЛЕЙ (Target: 60-70% accuracy)")
    logger.info("=" * 80)
    
    results = []
    
    for symbol in SYMBOLS:
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"📊 Тренування покращеної моделі для {symbol}")
            logger.info(f"{'=' * 80}\n")
            
            trainer = AdvancedModelTrainer(symbol, INTERVAL)
            
            # Завантаження даних
            data = await trainer.load_data()
            if data is None:
                continue
            
            # Підготовка фічей
            df = trainer.prepare_features(data)
            
            # Створення послідовностей
            X, y = trainer.create_sequences(df)
            
            # Тренування
            result = trainer.train(X, y)
            results.append(result)
            
            logger.info(f"✅ {symbol} тренування завершено! "
                       f"Accuracy: {result['val_dir_acc']:.2%}\n")
            
        except Exception as e:
            logger.error(f"❌ Помилка тренування {symbol}: {e}", exc_info=True)
            continue
    
    # Підсумок
    logger.info("\n" + "=" * 80)
    logger.info("📊 ПІДСУМОК ПОКРАЩЕНОГО ТРЕНУВАННЯ")
    logger.info("=" * 80)
    
    for result in sorted(results, key=lambda x: x['val_dir_acc'], reverse=True):
        emoji = "🎯" if result['val_dir_acc'] >= 0.60 else "✅" if result['val_dir_acc'] >= 0.55 else "⚠️"
        logger.info(f"{emoji} {result['symbol']:12} | Accuracy: {result['val_dir_acc']:.2%} | "
                   f"Loss: {result['val_loss']:.6f}")
    
    avg_acc = sum(r['val_dir_acc'] for r in results) / len(results)
    logger.info(f"\n📈 Середня accuracy: {avg_acc:.2%}")
    logger.info(f"✅ Натреновано {len(results)}/{len(SYMBOLS)} моделей")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(train_all_models_advanced())
