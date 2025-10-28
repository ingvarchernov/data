#!/usr/bin/env python3
"""
Batch Training для Simple Trend Classifier (Random Forest)
Тренування окремої моделі для кожної валюти
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.simple_trend_classifier import SimpleTrendClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Валюти для тренування
SYMBOLS = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'SOLUSDT',
    'ADAUSDT',
    'DOGEUSDT',
    'XRPUSDT',
]

# Конфігурація
CONFIG = {
    'timeframe': '4h',  # 4h має кращу точність ніж 1h або 1d
    'days': 730,        # 2 роки історії
}


async def train_all_symbols():
    """
    Тренування моделей для всіх валют
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 BATCH TRAINING: Random Forest Trend Classifier")
    logger.info("="*80 + "\n")
    
    results = {}
    start_time = datetime.now()
    
    for symbol in SYMBOLS:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Тренування: {symbol}")
        logger.info(f"{'='*80}\n")
        
        try:
            trainer = SimpleTrendClassifier(
                symbol=symbol,
                timeframe=CONFIG['timeframe']
            )
            
            result = await trainer.train(days=CONFIG['days'])
            results[symbol] = {
                'status': 'success',
                'test_accuracy': result['test_accuracy'],
                'duration': result['duration'],
                'model_path': result['model_path']
            }
            
            # Виводимо результат
            acc = result['test_accuracy'] * 100
            status = "✅ SUCCESS" if acc >= 70 else "⚠️ NEEDS IMPROVEMENT"
            logger.info(f"\n{status}: {symbol} - Accuracy: {acc:.2f}%\n")
            
        except Exception as e:
            logger.error(f"❌ {symbol} FAILED: {e}")
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Підсумок
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("📊 BATCH TRAINING SUMMARY")
    logger.info("="*80 + "\n")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    logger.info(f"Completed: {success_count}/{len(SYMBOLS)}")
    logger.info(f"Total time: {duration:.1f}s ({duration/60:.1f}m)\n")
    
    # Детальні результати
    logger.info("Results by symbol:")
    for symbol, result in results.items():
        if result['status'] == 'success':
            acc = result['test_accuracy'] * 100
            emoji = "🎉" if acc >= 70 else "⚠️"
            logger.info(f"  {emoji} {symbol:12s} {acc:6.2f}%  ({result['duration']:.1f}s)")
        else:
            logger.info(f"  ❌ {symbol:12s} FAILED: {result.get('error', 'Unknown')}")
    
    # Збереження результатів
    log_dir = Path('logs/batch_training')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = log_dir / f'rf_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': CONFIG,
            'symbols': SYMBOLS,
            'results': results,
            'summary': {
                'success_count': success_count,
                'total_symbols': len(SYMBOLS),
                'duration_seconds': duration
            }
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved: {results_file}\n")
    
    # Повертаємо результати для використання в інших скриптах
    return [
        {
            'symbol': symbol,
            'success': result['status'] == 'success',
            'test_accuracy': result.get('test_accuracy', 0),
            'training_time': result.get('duration', 0),
            'error': result.get('error')
        }
        for symbol, result in results.items()
    ]


if __name__ == '__main__':
    asyncio.run(train_all_symbols())
