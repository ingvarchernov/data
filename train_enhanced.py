#!/usr/bin/env python3
"""
🚀 ENHANCED TRAINING - Покращене тренування для досягнення 60-70% accuracy
Комплексний підхід з усіма найкращими практиками ML для крипто-прогнозування
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_config import configure_gpu
from optimized_model import OptimizedPricePredictionModel
from train_model_advanced import AdvancedModelTrainer
from intelligent_sys import UnifiedBinanceLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

configure_gpu()

# ============================================================================
# 🎯 ПОКРАЩЕНА КОНФІГУРАЦІЯ ДЛЯ 60-70% ACCURACY
# ============================================================================

ENHANCED_CONFIG = {
    # === АРХІТЕКТУРА ===
    'model_type': 'advanced_lstm',
    'sequence_length': 120,  # ⬆️ Збільшено з 60 до 120 (5 днів контексту)
    'batch_size': 32,        # ⬇️ Зменшено для кращого навчання
    'epochs': 500,           # ⬆️ Збільшено з 200 до 500
    'learning_rate': 0.0001, # ⬇️ Менша LR для точнішого навчання
    'early_stopping_patience': 50,  # ⬆️ Більше терпіння
    'reduce_lr_patience': 15,
    
    # === LSTM LAYERS ===
    'lstm_units_1': 512,     # ⬆️ Збільшено з 320
    'lstm_units_2': 256,     # ⬆️ Збільшено з 160
    'lstm_units_3': 128,     # ⬆️ Збільшено з 80
    'lstm_units_4': 64,      # ➕ Додатковий шар
    
    # === ATTENTION ===
    'attention_heads': 16,   # ⬆️ Збільшено з 10
    'attention_key_dim': 128, # ⬆️ Збільшено з 80
    
    # === DENSE LAYERS ===
    'dense_units': [1024, 512, 256, 128, 64],  # ⬆️ Глибша мережа
    
    # === REGULARIZATION ===
    'dropout_rate': 0.35,    # ⬇️ Трохи менше для складнішої моделі
    'l2_regularization': 0.005,  # ⬇️ Менше для збільшеної мережі
    
    # === DATA ===
    'days_back': 730,        # ⬆️ 2 роки історії замість 1
    'validation_split': 0.15,
    'test_split': 0.10,
}

# ============================================================================
# 🔧 РОЗШИРЕНИЙ FEATURE ENGINEERING
# ============================================================================

ENHANCED_FEATURES = {
    # === БАЗОВІ ТЕХНІЧНІ ІНДИКАТОРИ ===
    'basic': [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns',
    ],
    
    # === ТРЕНД ===
    'trend': [
        'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
        'sma_9', 'sma_21', 'sma_50', 'sma_100', 'sma_200',
        'tema_9', 'tema_21',  # Triple EMA
        'dema_9', 'dema_21',  # Double EMA
        'kama_10', 'kama_30',  # Kaufman Adaptive MA
    ],
    
    # === MOMENTUM ===
    'momentum': [
        'rsi_7', 'rsi_14', 'rsi_21',
        'roc_5', 'roc_10', 'roc_20',
        'mom_5', 'mom_10', 'mom_20',
        'tsi',  # True Strength Index
        'uo',   # Ultimate Oscillator
        'ppo',  # Percentage Price Oscillator
        'williams_r_14',
        'cci_20',
        'stoch_k', 'stoch_d',
        'stochrsi_k', 'stochrsi_d',
    ],
    
    # === VOLATILITY ===
    'volatility': [
        'atr_7', 'atr_14', 'atr_21',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
        'kc_upper', 'kc_middle', 'kc_lower',  # Keltner Channels
        'dc_upper', 'dc_middle', 'dc_lower',  # Donchian Channels
        'hvol_10', 'hvol_20', 'hvol_30',  # Historical Volatility
        'natr',  # Normalized ATR
        'true_range',
    ],
    
    # === VOLUME ===
    'volume': [
        'obv',  # On-Balance Volume
        'ad',   # Accumulation/Distribution
        'adosc',  # Chaikin A/D Oscillator
        'cmf',  # Chaikin Money Flow
        'mfi',  # Money Flow Index
        'eom',  # Ease of Movement
        'vwap',
        'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
        'volume_ratio',
        'volume_momentum',
    ],
    
    # === MACD ===
    'macd': [
        'macd', 'macd_signal', 'macd_histogram',
        'macd_diff',
    ],
    
    # === ICHIMOKU ===
    'ichimoku': [
        'ichimoku_a', 'ichimoku_b',
        'ichimoku_base', 'ichimoku_conv',
    ],
    
    # === PRICE ACTION ===
    'price_action': [
        'body', 'upper_wick', 'lower_wick',
        'candle_size',
        'high_low_range',
        'close_position',  # Позиція close в high-low range
    ],
    
    # === СТАТИСТИЧНІ ФІЧІ ===
    'stats': [
        'close_std_5', 'close_std_10', 'close_std_20', 'close_std_30',
        'close_skew_20', 'close_kurt_20',
        'returns_std_20',
        'rolling_sharpe_20',
    ],
    
    # === ДОДАТКОВІ РОЗРАХОВАНІ ФІЧІ ===
    'derived': [
        'price_momentum',
        'acceleration',
        'dist_from_ema_21', 'dist_from_ema_50', 'dist_from_ema_200',
        'trend_strength',
        'volatility_trend',
    ],
}


class EnhancedFeatureEngineer:
    """Розширений feature engineering з усіма можливими індикаторами"""
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок всіх технічних індикаторів"""
        logger.info("🔧 Розрахунок розширених фічей...")
        
        data = df.copy()
        
        # === БАЗОВІ ===
        data['returns'] = data['close'].pct_change() * 100
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1)) * 100
        
        # === ТРЕНД - EMA ===
        for period in [9, 21, 50, 100, 200]:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # === ТРЕНД - SMA ===
        for period in [9, 21, 50, 100, 200]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        # === MOMENTUM - RSI ===
        for period in [7, 14, 21]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # === MOMENTUM - ROC ===
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / 
                                     data['close'].shift(period)) * 100
        
        # === MOMENTUM - MOM ===
        for period in [5, 10, 20]:
            data[f'mom_{period}'] = data['close'].diff(period)
        
        # === MOMENTUM - Williams %R ===
        period = 14
        high_roll = data['high'].rolling(window=period).max()
        low_roll = data['low'].rolling(window=period).min()
        data['williams_r_14'] = -100 * ((high_roll - data['close']) / (high_roll - low_roll + 1e-10))
        
        # === MOMENTUM - CCI ===
        period = 20
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        data['cci_20'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # === MOMENTUM - Stochastic ===
        period = 14
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min + 1e-10))
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # === VOLATILITY - ATR ===
        for period in [7, 14, 21]:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data[f'atr_{period}'] = tr.rolling(window=period).mean()
        
        # === VOLATILITY - Bollinger Bands ===
        period = 20
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        data['bb_upper'] = sma + (2 * std)
        data['bb_middle'] = sma
        data['bb_lower'] = sma - (2 * std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
        
        # === VOLATILITY - Historical Volatility ===
        for period in [10, 20, 30]:
            data[f'hvol_{period}'] = data['returns'].rolling(window=period).std() * np.sqrt(24)
        
        # === VOLUME ===
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        for period in [5, 10, 20]:
            data[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
        
        data['volume_ratio'] = data['volume'] / (data['volume_sma_20'] + 1e-10)
        data['volume_momentum'] = data['volume'].pct_change(5) * 100
        
        # === VOLUME - MFI ===
        period = 14
        tp = (data['high'] + data['low'] + data['close']) / 3
        mf = tp * data['volume']
        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
        data['mfi'] = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))
        
        # === MACD ===
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        data['macd_diff'] = data['macd'].diff()
        
        # === PRICE ACTION ===
        data['body'] = np.abs(data['close'] - data['open'])
        data['upper_wick'] = data['high'] - np.maximum(data['close'], data['open'])
        data['lower_wick'] = np.minimum(data['close'], data['open']) - data['low']
        data['candle_size'] = data['high'] - data['low']
        data['high_low_range'] = data['high'] - data['low']
        data['close_position'] = (data['close'] - data['low']) / (data['high_low_range'] + 1e-10)
        
        # === СТАТИСТИЧНІ ===
        for period in [5, 10, 20, 30]:
            data[f'close_std_{period}'] = data['close'].rolling(window=period).std()
        
        data['close_skew_20'] = data['close'].rolling(window=20).skew()
        data['close_kurt_20'] = data['close'].rolling(window=20).kurt()
        data['returns_std_20'] = data['returns'].rolling(window=20).std()
        
        # === ДОДАТКОВІ ===
        data['price_momentum'] = data['close'].diff(5)
        data['acceleration'] = data['price_momentum'].diff()
        
        for period in [21, 50, 200]:
            data[f'dist_from_ema_{period}'] = (data['close'] - data[f'ema_{period}']) / data[f'ema_{period}'] * 100
        
        logger.info(f"✅ Розраховано {len(data.columns)} фічей")
        
        return data


class EnhancedModelTrainer:
    """Покращений trainer для досягнення 60-70% accuracy"""
    
    def __init__(self, symbol: str, interval: str = '1h'):
        self.symbol = symbol
        self.interval = interval
        self.config = ENHANCED_CONFIG.copy()
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 ENHANCED TRAINING - {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Конфігурація:")
        logger.info(f"   Model: {self.config['model_type']}")
        logger.info(f"   Sequence: {self.config['sequence_length']} (5 днів)")
        logger.info(f"   Batch: {self.config['batch_size']}")
        logger.info(f"   Epochs: {self.config['epochs']}")
        logger.info(f"   Learning Rate: {self.config['learning_rate']}")
        logger.info(f"   Data: {self.config['days_back']} днів (2 роки)")
        logger.info(f"   LSTM: {self.config['lstm_units_1']}-{self.config['lstm_units_2']}-{self.config['lstm_units_3']}-{self.config['lstm_units_4']}")
        logger.info(f"   Attention: {self.config['attention_heads']} heads, {self.config['attention_key_dim']} key_dim")
        logger.info(f"   Dense: {self.config['dense_units']}")
    
    async def load_data(self) -> pd.DataFrame:
        """Завантаження даних"""
        days = self.config['days_back']
        logger.info(f"\n📥 Завантаження {days} днів даних...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
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
                logger.error(f"❌ Недостатньо даних")
                return None
            
            logger.info(f"✅ Завантажено {len(data)} записів")
            return data
            
        finally:
            await loader.close()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Підготовка розширених фічей"""
        logger.info(f"\n🔧 Feature Engineering...")
        
        # Розрахунок всіх фічей
        df = EnhancedFeatureEngineer.calculate_all_features(data)
        
        # Видаляємо колонки з занадто багатьма NaN
        threshold = 0.3  # максимум 30% NaN
        for col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > threshold:
                logger.warning(f"⚠️ Видаляємо {col}: {nan_ratio*100:.1f}% NaN")
                df = df.drop(columns=[col])
        
        # Заповнюємо залишкові NaN
        initial_len = len(df)
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"✅ Підготовлено {len(df)} записів з {len(df.columns)} фічами")
        logger.info(f"   Видалено {dropped} записів з NaN")
        
        return df
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Створення послідовностей з target = % зміна ціни"""
        logger.info(f"\n🔄 Створення послідовностей...")
        
        sequence_length = self.config['sequence_length']
        
        # Всі колонки крім тих, що використовуємо для target
        feature_cols = [col for col in data.columns if col not in ['close', 'target']]
        
        # Target: відсоткова зміна ціни (наступна година)
        # Зберігаємо close для розрахунку target
        close_prices = data['close'].values
        
        # Нормалізація фічей
        scaled_data = self.scaler.fit_transform(data[feature_cols].values)
        
        X, y = [], []
        
        for i in range(len(scaled_data) - sequence_length):
            # Послідовність фічей
            X.append(scaled_data[i:i + sequence_length])
            
            # Target: % зміна ціни від поточної до наступної
            current_price = close_prices[i + sequence_length - 1]
            next_price = close_prices[i + sequence_length]
            pct_change = ((next_price - current_price) / current_price) * 100
            
            y.append(pct_change)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(f"✅ Створено послідовності:")
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   y shape: {y.shape}")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Target: відсоткова зміна ціни")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Тренування моделі"""
        logger.info(f"\n🎓 Початок тренування...")
        
        # Розділення на train/val/test
        val_size = int(len(X) * self.config['validation_split'])
        test_size = int(len(X) * self.config['test_split'])
        train_size = len(X) - val_size - test_size
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"📊 Розподіл даних:")
        logger.info(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"   Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Створення моделі
        n_features = X.shape[2]
        
        model = OptimizedPricePredictionModel(
            n_features=n_features,
            **self.config
        )
        
        self.model = model.build_model()
        
        # Callbacks
        model_dir = f'models/enhanced_{self.symbol.replace("USDT", "")}'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = f'{model_dir}/model_{timestamp}.keras'
        
        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_directional_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_directional_accuracy',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(f'{model_dir}/training_{timestamp}.csv'),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/tensorboard_{timestamp}',
                histogram_freq=1
            ),
        ]
        
        # Тренування
        logger.info(f"\n🏋️ Тренування моделі...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
        
        # Оцінка на тестовому наборі
        logger.info(f"\n📈 Оцінка на тестовому наборі...")
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        test_metrics = dict(zip(self.model.metrics_names, test_results))
        
        # Збереження scaler
        scaler_path = f'{model_dir}/scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Результати
        best_val_loss = min(history.history['val_loss'])
        best_val_dir_acc = max(history.history.get('val_directional_accuracy', [0]))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ ТРЕНУВАННЯ ЗАВЕРШЕНО")
        logger.info(f"{'='*80}")
        logger.info(f"\n📊 РЕЗУЛЬТАТИ:")
        logger.info(f"   Best val_loss: {best_val_loss:.6f}")
        logger.info(f"   Best val_directional_accuracy: {best_val_dir_acc:.4f} ({best_val_dir_acc*100:.2f}%)")
        logger.info(f"\n🧪 ТЕСТОВИЙ НАБІР:")
        for metric, value in test_metrics.items():
            if 'accuracy' in metric:
                logger.info(f"   {metric}: {value:.4f} ({value*100:.2f}%)")
            else:
                logger.info(f"   {metric}: {value:.6f}")
        logger.info(f"\n💾 ЗБЕРЕЖЕНО:")
        logger.info(f"   Model: {checkpoint_path}")
        logger.info(f"   Scaler: {scaler_path}")
        
        return {
            'symbol': self.symbol,
            'val_loss': best_val_loss,
            'val_dir_acc': best_val_dir_acc,
            'test_metrics': test_metrics,
            'model_path': checkpoint_path,
            'scaler_path': scaler_path,
        }


async def train_enhanced_btc():
    """Тренування enhanced моделі для BTC"""
    try:
        trainer = EnhancedModelTrainer('BTCUSDT')
        
        # Завантаження даних
        data = await trainer.load_data()
        if data is None:
            return None
        
        # Підготовка фічей
        df = trainer.prepare_features(data)
        
        # Створення послідовностей
        X, y = trainer.create_sequences(df)
        
        # Тренування
        result = trainer.train(X, y)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ ПОМИЛКА: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("🚀 ENHANCED TRAINING FOR BTC - Версія для досягнення 60-70% accuracy")
    logger.info("="*80 + "\n")
    
    result = asyncio.run(train_enhanced_btc())
    
    if result:
        logger.info("\n🎉 УСПІХ!")
        sys.exit(0)
    else:
        logger.error("\n❌ НЕВДАЧА")
        sys.exit(1)
