#!/usr/bin/env python3
"""
Перевірка accuracy всіх натренованих моделей
"""
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_all_models():
    """Перевіряємо всі моделі"""
    models_dir = Path('models')
    
    results = []
    
    for model_dir in sorted(models_dir.glob('simple_trend_*')):
        symbol = model_dir.name.replace('simple_trend_', '')
        
        # Шукаємо pkl файли
        pkl_files = list(model_dir.glob('model_*.pkl'))
        
        if not pkl_files:
            continue
        
        model_path = pkl_files[0]
        timeframe = model_path.stem.split('_')[-1]
        
        try:
            # Завантажуємо модель
            model = joblib.load(str(model_path))
            
            # Якщо є збережена accuracy
            if hasattr(model, 'test_accuracy'):
                accuracy = model.test_accuracy
            else:
                accuracy = None
            
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'accuracy': accuracy,
                'path': str(model_path)
            })
            
        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")
    
    # Виводимо результати
    logger.info("\n" + "="*80)
    logger.info("📊 НАТРЕНОВАНІ МОДЕЛІ")
    logger.info("="*80 + "\n")
    
    # Сортуємо за accuracy
    results_with_acc = [r for r in results if r['accuracy'] is not None]
    results_no_acc = [r for r in results if r['accuracy'] is None]
    
    results_with_acc.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if results_with_acc:
        for r in results_with_acc:
            acc = r['accuracy']
            
            if acc >= 0.70:
                mark = "✅"
            elif acc >= 0.60:
                mark = "⚠️"
            else:
                mark = "❌"
            
            logger.info(f"{mark} {r['symbol']:12s} [{r['timeframe']}] {acc:6.2%}")
    
    if results_no_acc:
        logger.info(f"\n⚠️ Моделі без збереженої accuracy: {len(results_no_acc)}")
        for r in results_no_acc:
            logger.info(f"   {r['symbol']:12s} [{r['timeframe']}]")
    
    # Статистика
    logger.info("\n" + "="*80)
    above_70 = [r for r in results_with_acc if r['accuracy'] >= 0.70]
    above_60 = [r for r in results_with_acc if 0.60 <= r['accuracy'] < 0.70]
    below_60 = [r for r in results_with_acc if r['accuracy'] < 0.60]
    
    logger.info(f"Всього моделей: {len(results)}")
    logger.info(f"✅ Accuracy ≥70%: {len(above_70)} - ГОТОВІ ДО ТОРГІВЛІ")
    logger.info(f"⚠️ Accuracy 60-70%: {len(above_60)} - обережно")
    logger.info(f"❌ Accuracy <60%: {len(below_60)} - не рекомендується")
    logger.info("="*80 + "\n")
    
    if above_70:
        symbols = [r['symbol'] for r in above_70]
        logger.info(f"🚀 Рекомендовані символи: {', '.join(symbols)}")


if __name__ == "__main__":
    check_all_models()
