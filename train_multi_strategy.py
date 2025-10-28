#!/usr/bin/env python3
"""
Тренування моделей для мультистратегійної системи:
- 4h таймфрейм: трендові стратегії
- 1h таймфрейм: свінг-стратегії

Додаємо більше символів для більше торгових можливостей
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.simple_trend_classifier import SimpleTrendClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ТОП-20 ліквідних криптовалют
ALL_SYMBOLS = [
    # Вже є 4h моделі
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'SOLUSDT',
    'ADAUSDT',
    'DOGEUSDT',
    'XRPUSDT',
    
    # Додаткові ліквідні пари
    'AVAXUSDT',
    'LINKUSDT',
    'MATICUSDT',
    'DOTUSDT',
    'UNIUSDT',
    'LTCUSDT',
    'ATOMUSDT',
    'ETCUSDT',
    'XLMUSDT',
    'ALGOUSDT',
    'VETUSDT',
    'FILUSDT',
    'TRXUSDT',
]

# Конфігурація тренування
TIMEFRAMES = {
    '4h': {
        'days': 730,  # 2 роки
        'description': 'Трендові стратегії'
    },
    '1h': {
        'days': 365,  # 1 рік (більше даних для коротших таймфреймів)
        'description': 'Свінг-стратегії'
    }
}


async def train_symbol(symbol: str, timeframe: str, days: int):
    """Тренування однієї моделі"""
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 {symbol} [{timeframe}]")
        logger.info(f"{'='*80}")
        
        trainer = SimpleTrendClassifier(
            symbol=symbol,
            timeframe=timeframe
        )
        
        result = await trainer.train(days=days)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': True,
            'test_accuracy': result['test_accuracy'],
            'training_time': result['duration'],
            'model_path': result['model_path']
        }
        
    except Exception as e:
        logger.error(f"❌ Помилка тренування {symbol} [{timeframe}]: {e}")
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': False,
            'error': str(e)
        }


async def train_multi_strategy_models(
    symbols: list = None,
    timeframes: list = None,
    skip_existing: bool = True
):
    """
    Тренування моделей для мультистратегійної системи
    
    Args:
        symbols: список символів (якщо None - всі)
        timeframes: список таймфреймів (якщо None - всі)
        skip_existing: пропускати вже натреновані моделі
    """
    
    if symbols is None:
        symbols = ALL_SYMBOLS
    
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    
    logger.info("\n" + "="*80)
    logger.info("🚀 MULTI-STRATEGY TRAINING")
    logger.info("="*80)
    logger.info(f"Символи: {len(symbols)}")
    logger.info(f"Таймфрейми: {timeframes}")
    logger.info(f"Всього моделей: {len(symbols) * len(timeframes)}")
    logger.info("="*80 + "\n")
    
    results = []
    start_time = datetime.now()
    
    for timeframe in timeframes:
        tf_config = TIMEFRAMES[timeframe]
        logger.info(f"\n{'#'*80}")
        logger.info(f"📈 {timeframe.upper()} ТАЙМФРЕЙМ - {tf_config['description']}")
        logger.info(f"{'#'*80}\n")
        
        for symbol in symbols:
            # Перевірка чи модель вже існує
            if skip_existing and timeframe == '4h':
                model_dir = Path(f'models/simple_trend_{symbol}')
                if model_dir.exists():
                    logger.info(f"⏭️ {symbol} [{timeframe}]: модель вже існує, пропускаємо")
                    continue
            
            result = await train_symbol(
                symbol=symbol,
                timeframe=timeframe,
                days=tf_config['days']
            )
            results.append(result)
    
    # Підсумок
    total_time = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("📊 РЕЗУЛЬТАТИ ТРЕНУВАННЯ")
    logger.info("="*80)
    
    # Групуємо по таймфреймах
    for timeframe in timeframes:
        tf_results = [r for r in results if r['timeframe'] == timeframe]
        successful = [r for r in tf_results if r.get('success')]
        failed = [r for r in tf_results if not r.get('success')]
        
        logger.info(f"\n{timeframe.upper()} ТАЙМФРЕЙМ:")
        logger.info(f"  Успішно: {len(successful)}/{len(tf_results)}")
        
        if successful:
            # Сортуємо за accuracy
            successful.sort(key=lambda x: x.get('test_accuracy', 0), reverse=True)
            
            for r in successful:
                acc = r.get('test_accuracy', 0)
                time_taken = r.get('training_time', 0)
                symbol = r['symbol']
                
                # Кольорове позначення
                if acc >= 0.70:
                    mark = "✅"
                elif acc >= 0.60:
                    mark = "⚠️"
                else:
                    mark = "❌"
                
                logger.info(f"  {mark} {symbol:12s} {acc:6.2%} ({time_taken:5.1f}s)")
        
        if failed:
            logger.info(f"\n  ❌ Помилки: {len(failed)}")
            for r in failed:
                logger.info(f"     {r['symbol']}: {r.get('error', 'Unknown')[:50]}")
    
    # Загальна статистика
    all_successful = [r for r in results if r.get('success')]
    above_70 = [r for r in all_successful if r.get('test_accuracy', 0) >= 0.70]
    above_60 = [r for r in all_successful if 0.60 <= r.get('test_accuracy', 0) < 0.70]
    below_60 = [r for r in all_successful if r.get('test_accuracy', 0) < 0.60]
    
    logger.info("\n" + "="*80)
    logger.info("🎯 ПІДСУМОК:")
    logger.info(f"  Всього моделей: {len(results)}")
    logger.info(f"  ✅ Accuracy ≥70%: {len(above_70)} (готові до торгівлі)")
    logger.info(f"  ⚠️ Accuracy 60-70%: {len(above_60)} (обережно)")
    logger.info(f"  ❌ Accuracy <60%: {len(below_60)} (не рекомендується)")
    logger.info(f"  ⏱️ Загальний час: {total_time:.1f}s")
    logger.info("="*80 + "\n")
    
    return results


async def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Тренування мультистратегійних моделей')
    parser.add_argument('--symbols', nargs='+', help='Список символів (default: всі)')
    parser.add_argument('--timeframes', nargs='+', choices=['4h', '1h'], help='Таймфрейми')
    parser.add_argument('--no-skip', action='store_true', help='Перетренувати існуючі моделі')
    
    args = parser.parse_args()
    
    results = await train_multi_strategy_models(
        symbols=args.symbols,
        timeframes=args.timeframes,
        skip_existing=not args.no_skip
    )
    
    # Рекомендації
    good_models = [r for r in results if r.get('success') and r.get('test_accuracy', 0) >= 0.70]
    
    if good_models:
        logger.info("🚀 ГОТОВІ ДО ТОРГІВЛІ:")
        
        # Групуємо за таймфреймами
        for tf in ['4h', '1h']:
            tf_good = [r for r in good_models if r['timeframe'] == tf]
            if tf_good:
                symbols = [r['symbol'] for r in tf_good]
                logger.info(f"\n{tf.upper()}: {', '.join(symbols)}")


if __name__ == "__main__":
    asyncio.run(main())
