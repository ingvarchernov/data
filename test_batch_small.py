#!/usr/bin/env python3
"""Тест batch trainer на 3 валютах"""
import asyncio
import sys
sys.path.insert(0, '.')

from training.simple_trend_classifier import SimpleTrendClassifier

async def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)
        
        trainer = SimpleTrendClassifier(symbol=symbol, timeframe='4h')
        result = await trainer.train(days=730)
        
        acc = result['test_accuracy'] * 100
        print(f"\n✅ {symbol}: {acc:.2f}% accuracy")

if __name__ == '__main__':
    asyncio.run(main())
