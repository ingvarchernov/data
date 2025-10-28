#!/usr/bin/env python3
"""Простий запуск тренування без аргументів"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training.simple_trend_classifier import SimpleTrendClassifier

async def quick_train():
    """Швидке тренування основних валют"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    print("\n" + "="*80)
    print("🚀 ШВИДКЕ ТРЕНУВАННЯ МОДЕЛЕЙ")
    print("="*80)
    print(f"\n📊 Валют: {len(symbols)}")
    print(f"📅 Історія: 730 днів (2 роки)")
    print(f"⏱️  Час: ~{len(symbols) * 5} секунд\n")
    print("="*80 + "\n")
    
    results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 🎯 {symbol}")
        print("-" * 80)
        
        try:
            model_dir = Path(f"models/simple_trend_{symbol}")
            model_file = model_dir / f"model_{symbol}_4h.pkl"
            
            if model_file.exists():
                print(f"⏭️  Модель вже існує, пропускаю")
                results.append({'symbol': symbol, 'status': 'skipped'})
                continue
            
            classifier = SimpleTrendClassifier(symbol=symbol, interval='4h')
            
            print(f"📥 Завантаження даних...")
            await classifier.prepare_data(days=730)
            
            print(f"🤖 Тренування Random Forest...")
            metrics = await classifier.train()
            
            print(f"💾 Збереження моделі...")
            classifier.save_model()
            
            acc = metrics.get('test_accuracy', 0)
            print(f"✅ {symbol}: Test Accuracy = {acc:.2%}")
            
            results.append({
                'symbol': symbol,
                'status': 'success',
                'accuracy': acc
            })
            
        except Exception as e:
            print(f"❌ {symbol}: Помилка - {e}")
            results.append({
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            })
    
    # Підсумок
    print("\n" + "="*80)
    print("📊 ПІДСУМОК")
    print("="*80 + "\n")
    
    success = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors = [r for r in results if r['status'] == 'error']
    
    if success:
        print(f"✅ УСПІШНО: {len(success)}")
        for r in sorted(success, key=lambda x: x.get('accuracy', 0), reverse=True):
            print(f"   {r['symbol']:12} - {r['accuracy']:.2%}")
    
    if skipped:
        print(f"\n⏭️  ПРОПУЩЕНО: {len(skipped)}")
        for r in skipped:
            print(f"   {r['symbol']}")
    
    if errors:
        print(f"\n❌ ПОМИЛКИ: {len(errors)}")
        for r in errors:
            print(f"   {r['symbol']:12} - {r.get('error', 'Unknown')}")
    
    print(f"\n{'='*80}")
    print(f"Всього: {len(results)} | Успіх: {len(success)} | Пропущено: {len(skipped)} | Помилки: {len(errors)}")
    print(f"{'='*80}\n")
    
    if success:
        print("🎉 Готово! Тепер можете запустити бота:")
        print("   python simple_trading_bot.py --symbols", " ".join([r['symbol'] for r in success[:6]]), "--testnet")

if __name__ == "__main__":
    try:
        asyncio.run(quick_train())
    except KeyboardInterrupt:
        print("\n\n⚠️  Тренування зупинено користувачем")
    except Exception as e:
        print(f"\n\n❌ Критична помилка: {e}")
        import traceback
        traceback.print_exc()
