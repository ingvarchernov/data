#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тренування Random Forest моделей для криптовалют
"""
import asyncio
import sys
from training.batch_train_rf import train_all_symbols

async def main():
    """Головна функція"""
    print("🚀 Запуск тренування Random Forest моделей...")
    print("=" * 60)
    
    try:
        results = await train_all_symbols()
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТИ ТРЕНУВАННЯ:")
        print("=" * 60)
        
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        if successful:
            print(f"\n✅ Успішно натреновано: {len(successful)}")
            for result in successful:
                symbol = result['symbol']
                acc = result.get('test_accuracy', 0)
                time_taken = result.get('training_time', 0)
                print(f"  • {symbol}: {acc:.2%} accuracy ({time_taken:.1f}s)")
        
        if failed:
            print(f"\n❌ Помилки: {len(failed)}")
            for result in failed:
                symbol = result['symbol']
                error = result.get('error', 'Unknown error')
                print(f"  • {symbol}: {error}")
        
        print("\n" + "=" * 60)
        
        # Рекомендації
        above_70 = [r for r in successful if r.get('test_accuracy', 0) >= 0.70]
        below_70 = [r for r in successful if r.get('test_accuracy', 0) < 0.70]
        
        if above_70:
            print(f"\n🎯 Моделі з accuracy ≥70%: {len(above_70)}")
            print("   Готові до live trading!")
        
        if below_70:
            print(f"\n⚠️  Моделі з accuracy <70%: {len(below_70)}")
            print("   Потребують додаткової оптимізації")
        
        return 0 if not failed else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Переривання користувачем")
        return 130
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
