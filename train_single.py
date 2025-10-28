#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тренування одного символу
"""
import asyncio
import sys
import argparse
from training.simple_trend_classifier import SimpleTrendClassifier

async def main():
    parser = argparse.ArgumentParser(description="Тренування Random Forest для одного символу")
    parser.add_argument('symbol', type=str, help='Символ (наприклад, BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='4h', help='Таймфрейм (за замовч. 4h)')
    parser.add_argument('--days', type=int, default=730, help='Кількість днів історії (за замовч. 730)')
    
    args = parser.parse_args()
    
    print(f"🚀 Тренування {args.symbol} на {args.timeframe} таймфреймі...")
    print(f"📅 Історія: {args.days} днів")
    print("=" * 60)
    
    try:
        classifier = SimpleTrendClassifier(
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        
        print("📊 Підготовка даних...")
        await classifier.prepare_data(days=args.days)
        
        print("🤖 Тренування моделі...")
        metrics = await classifier.train()
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТИ:")
        print("=" * 60)
        print(f"Train accuracy: {metrics['train_accuracy']:.2%}")
        print(f"Test accuracy:  {metrics['test_accuracy']:.2%}")
        print(f"\nПрецизія DOWN: {metrics['precision_down']:.2f}")
        print(f"Прецизія UP:   {metrics['precision_up']:.2f}")
        print(f"\nRecall DOWN: {metrics['recall_down']:.2f}")
        print(f"Recall UP:   {metrics['recall_up']:.2f}")
        
        if metrics['test_accuracy'] >= 0.70:
            print("\n🎯 Модель готова до використання! (accuracy ≥70%)")
        else:
            print(f"\n⚠️  Потрібна оптимізація (accuracy {metrics['test_accuracy']:.2%} < 70%)")
        
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Переривання користувачем")
        return 130
    except Exception as e:
        print(f"\n❌ Помилка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
