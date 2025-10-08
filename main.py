# -*- coding: utf-8 -*-
"""
Оптимізований головний модуль з новою архітектурою
Інтегрує всі оптимізації: асинхронність, кешування, GPU, Rust індикатори
"""
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
from optimized_model import OptimizedPricePredictionModel
from cache_system import cache_manager, get_cache_info
from async_architecture import ml_pipeline, init_async_system, shutdown_async_system
from gpu_config import configure_gpu, get_gpu_info
from config import SYMBOL, INTERVAL, DAYS_BACK, LOOK_BACK, STEPS

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
                                     force_retrain: bool = False):
        """Оптимізована обробка символу"""
        logger.info(f"📈 Початок обробки {symbol} ({interval})")
        start_time = datetime.now()
        
        try:
            # 1. Завантаження та підготовка даних
            logger.info("📊 Завантаження історичних даних...")
            symbol_id = await db_manager.get_or_create_symbol_id(symbol)
            interval_id = await db_manager.get_or_create_interval_id(interval)
            
            # Отримання даних з кешуванням
            data = await db_manager.get_historical_data_optimized(
                symbol_id, interval_id, days_back, use_cache=True
            )
            
            if data.empty:
                logger.error(f"❌ Немає даних для {symbol}")
                return None
            
            logger.info(f"📊 Завантажено {len(data)} записів")
            
            # 2. Розрахунок технічних індикаторів (асинхронно)
            logger.info("🔧 Розрахунок технічних індикаторів...")
            indicators = await global_calculator.calculate_all_indicators_batch(data)
            
            # Додавання індикаторів до даних
            for name, indicator in indicators.items():
                if len(indicator) > 0:
                    # Вирівнюємо індекси, уникаємо дублювання колонок
                    data = data.join(indicator, how='inner', lsuffix='_orig', rsuffix=f'_{name}')
            
            # Очищення від NaN
            data = data.dropna()
            logger.info(f"📊 Після додавання індикаторів: {len(data)} записів")
            
            if len(data) < look_back:
                logger.error(f"❌ Недостатньо даних після обробки: {len(data)} < {look_back}")
                return None
            
            # 3. Підготовка стратегічних фічей для ML
            # Додаємо стратегічні фічі
            data['trend'] = data['close'] - data['EMA_20'] if 'EMA_20' in data.columns else 0
            data['volatility'] = data['ATR'] if 'ATR' in data.columns else 0
            data['return'] = data['close'].pct_change().fillna(0)
            data['momentum'] = data['close'] - data['close'].shift(5).fillna(0)
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                data['bb_dist_upper'] = data['BB_Upper'] - data['close']
                data['bb_dist_lower'] = data['close'] - data['BB_Lower']
            # Додаємо всі технічні індикатори
            strategic_features = ['trend', 'volatility', 'return', 'momentum', 'bb_dist_upper', 'bb_dist_lower',
                                  'RSI', 'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D', 'ATR', 'EMA_20', 'BB_Upper', 'BB_Lower']
            # Додаємо об'єм
            if 'volume' in data.columns:
                strategic_features.append('volume')
            # Формуємо фінальний список фічей
            feature_columns = [col for col in data.columns if col not in ['timestamp', 'data_id']]
            # Додаємо стратегічні фічі, якщо їх немає
            for f in strategic_features:
                if f in data.columns and f not in feature_columns:
                    feature_columns.append(f)
            X_data = data[feature_columns].values

            # Нормалізація
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)

            # Створення послідовностей
            X_sequences = []
            for i in range(len(X_scaled) - look_back):
                X_sequences.append(X_scaled[i:i + look_back])
            X_sequences = np.array(X_sequences)

            if len(X_sequences) == 0:
                logger.error("❌ Не вдалося створити послідовності")
                return None
            
            # 4. Тренування/завантаження моделі
            model_path = f"models/optimized_{symbol}_{interval}.keras"
            
            if force_retrain or not Path(model_path).exists():
                logger.info("🤖 Тренування нової моделі...")
                
                # Створення моделі
                model_builder = OptimizedPricePredictionModel(
                    input_shape=(look_back, len(feature_columns)),
                    model_type="transformer_lstm"
                )
                
                # Поділ на тренувальну та валідаційну вибірки
                split_idx = int(len(X_sequences) * 0.8)
                X_train = X_sequences[:split_idx]
                X_val = X_sequences[split_idx:]
                
                # Цільові значення (прогноз наступної ціни)
                y_train = X_scaled[look_back:split_idx + look_back, feature_columns.index('close')]
                y_val = X_scaled[split_idx + look_back:, feature_columns.index('close')]
                
                # Явне приведення типів для CuDNN
                X_train = X_train.astype(np.float32)
                X_val = X_val.astype(np.float32)
                y_train = y_train.astype(np.float32)
                y_val = y_val.astype(np.float32)
                # Тренування
                model, history = model_builder.train_model(
                    X_train, y_train, X_val, y_val,
                    model_save_path=model_path,
                    epochs=100,
                    batch_size=64, 
                    learning_rate=0.001
                )
                
                # Збереження метаданих
                metadata = {
                    'symbol': symbol,
                    'interval': interval,
                    'features': feature_columns,
                    'scaler_params': {
                        'mean': scaler.mean_.tolist(),
                        'scale': scaler.scale_.tolist()
                    },
                    'trained_at': datetime.now().isoformat(),
                    'data_shape': X_sequences.shape,
                    'model_type': 'transformer_lstm'
                }
                
                model_builder.save_model_with_metadata(model, model_path, metadata)
                
            else:
                logger.info("📥 Завантаження існуючої моделі...")
                model_builder = OptimizedPricePredictionModel(
                    input_shape=(look_back, len(feature_columns)),
                    model_type="transformer_lstm"
                )
                model = model_builder.load_model_with_custom_objects(model_path)
            
            # 5. Прогнозування
            logger.info("🔮 Генерація прогнозів...")
            
            # Беремо останні дані для прогнозування
            last_sequence = X_scaled[-look_back:].reshape(1, look_back, len(feature_columns))
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for step in range(steps):
                pred = model.predict(current_sequence, verbose=0)
                predictions.append(float(pred[0, 0].item()))
                # Оновлюємо послідовність для наступного прогнозу
                new_row = current_sequence[0, -1, :].copy()
                new_row[feature_columns.index('close')] = float(pred[0, 0].item())
                # Зсуваємо послідовність
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_row
            
            # Денормалізація прогнозів
            # Денормалізуємо кожен прогноз окремо, підставляючи його у відповідний вектор фічей
            last_row = X_data[-1]
            predictions_denorm = []
            for p in predictions:
                denorm_vec = last_row.copy()
                denorm_vec[feature_columns.index('close')] = p
                denorm = scaler.inverse_transform([denorm_vec])[0][feature_columns.index('close')]
                predictions_denorm.append(denorm)
            
            # 6. Збереження результатів
            results = {
                'symbol': symbol,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                # Денормалізуємо останню ціну через scaler
                'last_price': scaler.inverse_transform([X_data[-1]])[0][feature_columns.index('close')],
                'predictions': predictions_denorm.tolist(),
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
    parser.add_argument("--look_back", type=int, default=LOOK_BACK, help="Розмір вікна")
    parser.add_argument("--steps", type=int, default=STEPS, help="Кроків прогнозу")
    parser.add_argument("--force_retrain", action="store_true", help="Примусове перетренування")
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
                force_retrain=args.force_retrain
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
                force_retrain=args.force_retrain
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