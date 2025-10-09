# -*- coding: utf-8 -*-
"""
Оптимізований головний модуль з новою архітектурою
Інтегрує всі оптимізації: асинхронність, кешування, GPU, Rust індикатори
"""
import numpy as np
import pandas as pd
import asyncio
import logging
import sys
import os
import argparse
from binance_loader import save_ohlcv_to_db
from datetime import datetime
from pathlib import Path

# Системні модулі
from dotenv import load_dotenv

# Оптимізовані модулі
from optimized_db import db_manager
from optimized_indicators import global_calculator
from optimized_model import OptimizedPricePredictionModel, DatabaseHistoryCallback, DenormalizedMetricsCallback
from cache_system import cache_manager, get_cache_info
from async_architecture import ml_pipeline, init_async_system, shutdown_async_system
from gpu_config import configure_gpu, get_gpu_info
# Використовуємо optimized_config замість config
from optimized_config import SYMBOL, INTERVAL, DAYS_BACK, LOOK_BACK, STEPS, MODEL_CONFIG

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OptimizedCryptoMLSystem:
    """Оптимізована система прогнозування криптовалют"""
    
    def __init__(self):
        self.initialized = False
        self.gpu_available = False
        
    async def initialize(self):
        """Ініціалізація системи"""
        if self.initialized:
            return
            
        logger.info("🚀 Ініціалізація оптимізованої системи...")
        
        # Завантаження змінних середовища
        load_dotenv()
        self._validate_environment()
        
        # Налаштування GPU
        self.gpu_available = configure_gpu()
        if self.gpu_available:
            gpu_info = get_gpu_info()
            logger.info(f"🔥 GPU статус: {gpu_info}")
        
        # Ініціалізація асинхронної системи
        await init_async_system()
        
        # Ініціалізація кешу
        logger.info("💾 Ініціалізація системи кешування...")
        cache_stats = get_cache_info()
        logger.info(f"📊 Статистика кешу: {cache_stats}")
        
        # Тестування з'єднання з БД
        try:
            await db_manager.execute_query_cached("SELECT 1 as test", use_cache=False)
            logger.info("✅ З'єднання з базою даних успішне")
        except Exception as e:
            logger.error(f"❌ Помилка з'єднання з БД: {e}")
            raise
        
        self.initialized = True
        logger.info("✅ Система ініціалізована успішно")
    
    def _validate_environment(self):
        """Валідація змінних середовища"""
        required_vars = ['API_KEY', 'API_SECRET', 'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Відсутні змінні середовища: {', '.join(missing_vars)}")
            raise ValueError(f"Необхідно визначити змінні: {', '.join(missing_vars)}")
    
    async def process_symbol_optimized(self, 
                                     symbol: str, 
                                     interval: str, 
                                     days_back: int,
                                     look_back: int,
                                     steps: int,
                                     force_retrain: bool = False,
                                     use_cv: bool = False):
        """Оптимізована обробка символу"""
        logger.info(f"📈 Початок обробки {symbol} ({interval})")
        start_time = datetime.now()
        
        try:
            # Кешування symbol_id та interval_id
            cache_key = f"{symbol}_{interval}_ids"
            cached_ids = cache_manager.get(cache_key)
            
            if cached_ids:
                symbol_id, interval_id = cached_ids
            else:
                symbol_id = await db_manager.get_or_create_symbol_id(symbol)
                interval_id = await db_manager.get_or_create_interval_id(interval)
                cache_manager.set(cache_key, (symbol_id, interval_id), ttl=86400)  # 24 години
            
            # Отримання даних з кешуванням
            data = await db_manager.get_historical_data_optimized(
                symbol_id, interval_id, days_back, use_cache=True
            )
            
            if data.empty:
                logger.error(f"❌ Немає даних для {symbol}")
                return None
            
            # Розрахунок технічних індикаторів (асинхронно)
            indicators = await global_calculator.calculate_all_indicators_batch(data)
            
            # Зберігаємо оригінальні OHLCV колонки перед join
            original_ohlcv = data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Додавання індикаторів до даних
            for name, indicator in indicators.items():
                if len(indicator) > 0:
                    data = data.join(indicator, how='inner', lsuffix='_orig', rsuffix=f'_{name}')
            
            # Відновлюємо оригінальні OHLCV колонки (join міг їх переписати)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in original_ohlcv.columns:
                    data[col] = original_ohlcv[col]
            
            # Очищення від NaN
            data = data.dropna()
            logger.info(f"✓ Обробка даних: {len(data)} записів після індикаторів")
            logger.info(f"📊 Після додавання індикаторів: {len(data)} записів")
            
            # ВИДАЛЕННЯ OUTLIERS - критичний крок для якості даних
            # Видаляємо екстремальні значення цін (більше 10 стандартних відхилень)
            price_cols = ['close', 'high', 'low', 'open']
            for col in price_cols:
                if col in data.columns:
                    mean_price = data[col].mean()
                    std_price = data[col].std()
                    # Видаляємо значення, які відхиляються більше ніж на 5 стандартних відхилень
                    data = data[abs(data[col] - mean_price) <= 5 * std_price]
            
            # Видаляємо екстремальні об'єми (більше 10 стандартних відхилень)
            if 'volume' in data.columns:
                vol_mean = data['volume'].mean()
                vol_std = data['volume'].std()
                data = data[abs(data['volume'] - vol_mean) <= 10 * vol_std]
            
            # Видаляємо екстремальні значення індикаторів
            indicator_cols = ['RSI', 'MACD', 'ATR', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'ADX']
            for col in indicator_cols:
                if col in data.columns:
                    # RSI має бути між 0-100, інші індикатори мають розумні межі
                    if col == 'RSI':
                        data = data[(data[col] >= 0) & (data[col] <= 100)]
                    elif col in ['Stoch_K', 'Stoch_D']:
                        data = data[(data[col] >= -20) & (data[col] <= 120)]  # Stochastic може виходити за 0-100
                    elif col == 'Williams_R':
                        data = data[(data[col] >= -100) & (data[col] <= 0)]  # Williams %R від -100 до 0
                    else:
                        # Для інших індикаторів видаляємо екстремальні значення
                        col_mean = data[col].mean()
                        col_std = data[col].std()
                        data = data[abs(data[col] - col_mean) <= 5 * col_std]
            
            logger.info(f"✓ Після видалення outliers: {len(data)} записів")
            
            if len(data) < look_back:
                logger.error(f"❌ Недостатньо даних після обробки: {len(data)} < {look_back}")
                return None
            
            # ДОДАТКОВІ СТАТИСТИЧНІ ФІЧІ
            # Rolling statistics для ціни
            data['close_rolling_mean_10'] = data['close'].rolling(10).mean()
            data['close_rolling_std_10'] = data['close'].rolling(10).std()
            data['close_rolling_skew_20'] = data['close'].rolling(20).skew()
            data['close_rolling_kurt_20'] = data['close'].rolling(20).kurt()
            
            # Volume-based features
            if 'volume' in data.columns:
                data['volume_rolling_mean_10'] = data['volume'].rolling(10).mean()
                data['volume_rolling_std_10'] = data['volume'].rolling(10).std()
                data['volume_to_price_ratio'] = data['volume'] / (data['close'] + 1e-6)
                data['volume_change'] = data['volume'].pct_change().fillna(0)
            
            # RSI-based features
            if 'RSI' in data.columns:
                data['rsi_overbought'] = (data['RSI'] > 70).astype(int)
                data['rsi_oversold'] = (data['RSI'] < 30).astype(int)
                data['rsi_divergence'] = data['RSI'].diff(5)  # 5-period RSI change
            
            # MACD-based features
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                data['macd_histogram'] = data['MACD'] - data['MACD_Signal']
                data['macd_crossover'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
                data['macd_trend'] = data['macd_histogram'].rolling(5).mean()
            
            # Bollinger Bands advanced features
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                data['bb_squeeze'] = (data['BB_Upper'] - data['BB_Lower']) / data['close']
                data['bb_breakout_up'] = (data['close'] > data['BB_Upper']).astype(int)
                data['bb_breakout_down'] = (data['close'] < data['BB_Lower']).astype(int)
            
            # Stochastic features
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                data['stoch_divergence'] = data['Stoch_K'] - data['Stoch_D']
                data['stoch_overbought'] = (data['Stoch_K'] > 80).astype(int)
                data['stoch_oversold'] = (data['Stoch_K'] < 20).astype(int)
            
            # ATR-based volatility features
            if 'ATR' in data.columns:
                data['atr_ratio'] = data['ATR'] / data['close']
                data['atr_change'] = data['ATR'].pct_change().fillna(0)
            
            # Price action patterns
            data['doji'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6) < 0.1
            data['hammer'] = ((data['high'] - data['low'] > 0) & 
                            (abs(data['open'] - data['close']) < 0.3 * (data['high'] - data['low'])) & 
                            ((data['low'] - data['close']) > 0.6 * (data['high'] - data['low']))).astype(int)
            
            # Time-based features (якщо є timestamp)
            if 'timestamp' in data.columns:
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                data['month'] = pd.to_datetime(data['timestamp']).dt.month
                # Циклічні features для часу
                data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
                data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
                data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            
            # Розширений список фічей для кращого навчання
            strategic_features = [
                # Basic OHLCV
                'close', 'volume', 'high', 'low', 'open',
                
                # Technical indicators
                'RSI', 'MACD', 'MACD_Signal', 'ATR', 'EMA_20', 'EMA_10', 'EMA_50',
                'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
                
                # Price-based features
                'trend', 'volatility', 'return', 'momentum', 'momentum_10', 'momentum_20',
                'return_5', 'return_10', 'close_lag1', 'close_lag2', 'close_diff', 'log_return',
                'close_rolling_mean_10', 'close_rolling_std_10', 'close_rolling_skew_20', 'close_rolling_kurt_20',
                
                # Volume features
                'volume_pct', 'volume_ma5', 'volume_ma20', 'volume_std',
                'volume_rolling_mean_10', 'volume_rolling_std_10', 'volume_to_price_ratio', 'volume_change',
                
                # Bollinger Bands
                'bb_dist_upper', 'bb_dist_lower', 'bb_width', 'bb_position', 'bb_squeeze', 'bb_breakout_up', 'bb_breakout_down',
                
                # RSI features
                'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
                
                # MACD features
                'macd_histogram', 'macd_crossover', 'macd_trend',
                
                # Stochastic features
                'stoch_divergence', 'stoch_overbought', 'stoch_oversold',
                
                # ATR features
                'atr_ratio', 'atr_change',
                
                # Price action patterns
                'doji', 'hammer', 'high_low_ratio', 'close_open_ratio',
                
                # Time features (if available)
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
            ]
            feature_columns = [f for f in strategic_features if f in data.columns]
            
            # Видаляємо NaN після lag-фічей та перевіряємо на Inf
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            
            if len(data) < look_back:
                logger.error(f"❌ Недостатньо даних після обробки: {len(data)} < {look_back}")
                return None
            
            logger.info(f"✓ Фінальний датасет: {len(data)} записів, {len(feature_columns)} фічей")
            
            # Time-series validation: використовуємо СЕРЕДНІ 20% як валідацію (не останні!)
            # Для фінансових даних краще використовувати дані з середини періоду
            # щоб уникнути проблем з різними ринковими умовами

            # Розділяємо на train/val/test: 60% / 20% / 20%
            n_total = len(data)
            train_end = int(n_total * 0.6)
            val_end = int(n_total * 0.8)

            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            # test_data = data.iloc[val_end:]  # зарезервовано для фінального тестування

            logger.info(f"📊 Хронологічний розподіл: Train={len(train_data)}, Val={len(val_data)} (60%/20%)")
            
            X_train_raw = train_data[feature_columns].values
            X_val_raw = val_data[feature_columns].values

            # ВАЖЛИВО: RobustScaler краще для trending фінансових даних
            # Використовує медіану та IQR замість min/max, стійкий до outliers
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)  # transform, НЕ fit_transform!
            
            # RobustScaler використовує center_ та scale_ замість min/max
            close_idx_feat = feature_columns.index('close')
            logger.info(f"✓ Scaler: train close center={scaler.center_[close_idx_feat]:.2f}, scale={scaler.scale_[close_idx_feat]:.2f}")

            # Створення послідовностей для train
            X_train_sequences = []
            for i in range(len(X_train_scaled) - look_back):
                X_train_sequences.append(X_train_scaled[i:i + look_back])
            X_train_sequences = np.array(X_train_sequences)
            
            # Створення послідовностей для val
            X_val_sequences = []
            for i in range(len(X_val_scaled) - look_back):
                X_val_sequences.append(X_val_scaled[i:i + look_back])
            X_val_sequences = np.array(X_val_sequences)
            
            # Цільові змінні - абсолютні ціни (close), а не різниці
            close_idx = feature_columns.index('close')
            y_train = X_train_scaled[look_back:, close_idx]  # наступні абсолютні ціни
            y_val = X_val_scaled[look_back:, close_idx]

            # Перевіряємо, чи співпадають розміри
            expected_train_len = len(X_train_scaled) - look_back
            expected_val_len = len(X_val_scaled) - look_back

            if len(y_train) != expected_train_len or len(y_val) != expected_val_len:
                logger.error(f"❌ Розмірність y не співпадає: y_train={len(y_train)}, expected={expected_train_len}")
                return None
            
            # Перевіряємо, чи співпадають розміри
            expected_train_len = len(X_train_scaled) - look_back
            expected_val_len = len(X_val_scaled) - look_back
            
            if len(y_train) != expected_train_len or len(y_val) != expected_val_len:
                logger.error(f"❌ Розмірність y не співпадає: y_train={len(y_train)}, expected={expected_train_len}")
                return None
            
            X_sequences = X_train_sequences  # Для сумісності з наступним кодом
            
            # Для prediction потрібні всі дані разом
            X_data = data[feature_columns].values
            X_all_scaled = scaler.transform(X_data)  # transform використовуючи train scaler

            if len(X_sequences) == 0:
                logger.error("❌ Не вдалося створити послідовності")
                return None
            
            # 4. Тренування/завантаження моделі

            model_path = f"models/optimized_{symbol}_{interval}.keras"

            retrain = force_retrain or not Path(model_path).exists()
            # Перевірка input_shape у метаданих
            metadata_path = model_path.replace('.keras', '_metadata.json')
            if not retrain and Path(metadata_path).exists():
                import json
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                old_shape = tuple(meta.get('data_shape', [0, 0]))
                new_shape = X_sequences.shape
                if old_shape != new_shape:
                    logger.info("⚠️ Input shape змінився, перетренування моделі...")
                    retrain = True

            if retrain:
                logger.info("🤖 Тренування нової моделі...")
                close_index = feature_columns.index('close')
                
                model_builder = OptimizedPricePredictionModel(
                    input_shape=(look_back, len(feature_columns)),
                    model_type="advanced_lstm",
                    scaler=scaler,
                    feature_index=close_index
                )
                
                # Використовуємо вже підготовлені X_train_sequences, X_val_sequences, y_train, y_val
                X_train = X_train_sequences.astype(np.float32)
                X_val = X_val_sequences.astype(np.float32)
                y_train = y_train.astype(np.float32)
                y_val = y_val.astype(np.float32)
                
                # Логування статистики цільової змінної
                logger.info(f"y_train: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}, std={y_train.std():.4f}")
                logger.info(f"y_val: min={y_val.min():.4f}, max={y_val.max():.4f}, mean={y_val.mean():.4f}, std={y_val.std():.4f}")
                
                # Перевірка на NaN/Inf
                if np.isnan(y_train).any() or np.isnan(y_val).any():
                    logger.error("❌ y_train або y_val містить NaN!")
                    return None
                if np.isinf(y_train).any() or np.isinf(y_val).any():
                    logger.error("❌ y_train або y_val містить Inf!")
                    return None
                
                # Створюємо callback для збереження в БД
                db_callback = DatabaseHistoryCallback(
                    db_engine=db_manager.sync_engine,
                    symbol_id=symbol_id,
                    interval_id=interval_id,
                    fold=1
                )
                
                # Створюємо callback для виводу денормалізованих метрик
                denorm_callback = DenormalizedMetricsCallback(
                    scaler=scaler,
                    feature_index=feature_columns.index('close'),
                    X_val=X_val,
                    y_val=y_val
                )
                
                # Використовуємо параметри з optimized_config
                model, history = model_builder.train_model(
                    X_train, y_train, X_val, y_val,
                    model_save_path=model_path,
                    epochs=MODEL_CONFIG['epochs'],
                    batch_size=MODEL_CONFIG['batch_size'],
                    learning_rate=MODEL_CONFIG['learning_rate'],
                    db_callback=db_callback,
                    additional_callbacks=[denorm_callback]
                )
                # Зберігаємо метадані з параметрами scaler
                scaler_params = {}
                if hasattr(scaler, 'data_min_'):
                    # MinMaxScaler
                    scaler_params = {
                        'type': 'MinMaxScaler',
                        'min': scaler.data_min_.tolist(),
                        'max': scaler.data_max_.tolist()
                    }
                elif hasattr(scaler, 'center_'):
                    # RobustScaler або StandardScaler
                    scaler_type = type(scaler).__name__
                    scaler_params = {
                        'type': scaler_type,
                        'center': scaler.center_.tolist(),
                        'scale': scaler.scale_.tolist()
                    }
                
                metadata = {
                    'symbol': symbol,
                    'interval': interval,
                    'features': feature_columns,
                    'scaler_params': scaler_params,
                    'trained_at': datetime.now().isoformat(),
                    'data_shape': X_sequences.shape,
                    'model_type': 'advanced_lstm'
                }
                model_builder.save_model_with_metadata(model, model_path, metadata)
            else:
                logger.info("📥 Завантаження існуючої моделі...")
                model_builder = OptimizedPricePredictionModel(
                    input_shape=(look_back, len(feature_columns)),
                    model_type="advanced_lstm",
                    scaler=scaler,
                    feature_index=feature_columns.index('close')
                )
                model = model_builder.load_model_with_custom_objects(model_path)
                
                # Якщо модель завантажена без компіляції, перекомпілюємо її
                if not model.compiled:
                    logger.info("🔧 Модель не скомпільована, компілюємо...")
                    model = model_builder.recompile_loaded_model(model)
            
            # 5. Прогнозування
            logger.info("🔮 Генерація прогнозів...")
            
            # Беремо останні дані для прогнозування
            last_sequence = X_all_scaled[-look_back:].reshape(1, look_back, len(feature_columns))
            
            predictions = []
            current_sequence = last_sequence.copy()

            for step in range(steps):
                # Модель передбачає нормалізовану абсолютну ціну
                pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]

                # Денормалізуємо передбачену ціну
                dummy = np.zeros((1, len(feature_columns)))
                dummy[0, close_idx] = pred_scaled
                predicted_price = scaler.inverse_transform(dummy)[0, close_idx]

                predictions.append(float(predicted_price))

                # Оновлюємо послідовність для наступного прогнозу
                new_row = current_sequence[0, -1, :].copy()
                new_row[close_idx] = pred_scaled  # Використовуємо нормалізовану передбачену ціну
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_row
            
            # Денормалізація прогнозів - просто використовуємо вже денормалізовані ціни
            predictions_denorm = predictions
            
            # 6. Збереження результатів
            # Денормалізуємо останню ціну (inverse scaler)
            last_scaled = scaler.inverse_transform([X_all_scaled[-1]])[0]
            last_price_denorm = last_scaled[feature_columns.index('close')]
            
            results = {
                'symbol': symbol,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'last_price': last_price_denorm,
                'predictions': predictions_denorm,
                'steps': steps,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Кешування результатів
            cache_key = f"predictions:{symbol}:{interval}:{steps}"
            cache_manager.set(cache_key, results, ttl=1800)
            
            logger.info(f"✅ Обробку завершено за {results['processing_time']:.2f}s")
            logger.info(f"📈 Остання ціна: {results['last_price']:.2f}")
            logger.info(f"🔮 Прогнози: {[f'{p:.2f}' for p in predictions_denorm]}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Помилка обробки {symbol}: {e}", exc_info=True)
            return None
    
    async def batch_process_symbols(self, symbols: list, **kwargs):
        """Пакетна обробка символів"""
        logger.info(f"🔄 Пакетна обробка {len(symbols)} символів")
        
        # Створюємо задачі для паралельного виконання
        tasks = []
        for symbol in symbols:
            task = self.process_symbol_optimized(symbol, **kwargs)
            tasks.append(task)
        
        # Виконуємо паралельно з обмеженням
        semaphore = asyncio.Semaphore(3)  # Максимум 3 символи одночасно
        
        async def limited_process(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_process(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Обробка результатів
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"✅ Пакетна обробка завершена: {successful} успішно, {failed} з помилками")
        
        return results
    
    async def get_system_status(self):
        """Отримання статусу системи"""
        return {
            'initialized': self.initialized,
            'gpu_available': self.gpu_available,
            'gpu_info': get_gpu_info() if self.gpu_available else None,
            'cache_stats': get_cache_info(),
            'worker_stats': ml_pipeline.worker_pool.get_stats() if ml_pipeline.worker_pool else None,
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Очищення ресурсів"""
        logger.info("🧹 Очищення ресурсів...")
        await shutdown_async_system()
        logger.info("✅ Очищення завершено")

# Глобальний екземпляр системи
crypto_system = OptimizedCryptoMLSystem()

async def main():
    """Головна асинхронна функція"""
    parser = argparse.ArgumentParser(description="Оптимізована система прогнозування криптовалют")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Торгова пара")
    parser.add_argument("--interval", type=str, default=INTERVAL, help="Інтервал часу")
    parser.add_argument("--days_back", type=int, default=DAYS_BACK, help="Днів історії")
    parser.add_argument("--look_back", type=int, default=LOOK_BACK, help="Розмір вікна з optimized_config")
    parser.add_argument("--steps", type=int, default=STEPS, help="Кроків прогнозу")
    parser.add_argument("--force_retrain", action="store_true", help="Примусове перетренування")
    parser.add_argument("--use_cv", action="store_true", help="Використати TimeSeriesSplit cross-validation")
    parser.add_argument("--batch", nargs="+", help="Пакетна обробка символів")
    parser.add_argument("--status", action="store_true", help="Показати статус системи")
    
    args = parser.parse_args()
    
    try:
        # Ініціалізація системи
        await crypto_system.initialize()

        # Автоматичне завантаження історичних даних з Binance
        logger.info("⏳ Завантаження історичних даних з Binance...")
        await save_ohlcv_to_db(db_manager, args.symbol, args.interval, days_back=args.days_back)
        logger.info("✅ Дані з Binance завантажено у historical_data")

        if args.status:
            # Показати статус
            status = await crypto_system.get_system_status()
            logger.info(f"📊 Статус системи: {status}")
            return
        
        if args.batch:
            # Пакетна обробка
            results = await crypto_system.batch_process_symbols(
                symbols=args.batch,
                interval=args.interval,
                days_back=args.days_back,
                look_back=args.look_back,
                steps=args.steps,
                force_retrain=args.force_retrain,
                use_cv=args.use_cv
            )
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Помилка обробки {args.batch[i]}: {result}")
                elif result:
                    logger.info(f"✅ {args.batch[i]}: {result['predictions']}")
        else:
            # Обробка одного символу
            result = await crypto_system.process_symbol_optimized(
                symbol=args.symbol,
                interval=args.interval,
                days_back=args.days_back,
                look_back=args.look_back,
                steps=args.steps,
                force_retrain=args.force_retrain,
                use_cv=args.use_cv
            )
            
            if result:
                logger.info("🎯 Результат прогнозування:")
                logger.info(f"   Символ: {result['symbol']}")
                logger.info(f"   Остання ціна: {result['last_price']:.2f}")
                logger.info(f"   Прогнози: {[f'{p:.2f}' for p in result['predictions']]}")
                logger.info(f"   Час обробки: {result['processing_time']:.2f}s")
    
    except KeyboardInterrupt:
        logger.info("⏸️ Отримано сигнал переривання")
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}", exc_info=True)
    finally:
        await crypto_system.cleanup()

if __name__ == "__main__":
    # Імпорт numpy для використання в функції
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Запуск головної функції
    asyncio.run(main())