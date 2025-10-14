#!/usr/bin/env python3
"""
Тестування швидкості індикаторів: Python vs Rust
"""
import time
import numpy as np
import pandas as pd
from binance_data_loader import BinanceDataLoader
from live_trading import LiveTradingSystem


def create_test_data(length=1000):
    """Створює тестові дані"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=length, freq='1H')

    # Генеруємо реалістичні ціни
    base_price = 50000
    prices = [base_price]
    for i in range(length - 1):
        change = np.random.normal(0, 0.02)  # 2% волатильність
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Мінімальна ціна

    # Створюємо OHLCV
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = [np.random.uniform(100, 1000) for _ in range(length)]

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

    return df


def benchmark_indicators():
    """Порівняння швидкості розрахунку індикаторів"""
    print("🚀 ТЕСТ ШВИДКОСТІ ІНДИКАТОРІВ")
    print("=" * 50)

    # Створюємо тестові дані
    df = create_test_data(5000)  # 5000 свічок
    print(f"📊 Тестові дані: {len(df)} свічок")

    # Ініціалізація системи
    system = LiveTradingSystem()

    # Тест швидкості
    iterations = 100
    print(f"🔄 Тестую {iterations} ітерацій розрахунку прогнозів...")

    start_time = time.time()
    for i in range(iterations):
        prediction = system._calculate_technical_prediction('BTCUSDT', df)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations

    print("✅ Результати тестування:")
    print(f"   Загальний час: {total_time:.3f} сек")
    print(f"   Середній час на ітерацію: {avg_time:.4f} сек")
    print(f"   Швидкість: {1/avg_time:.1f} прогнозів/сек")

    # Деталі прогнозу
    print("\n📈 Останній прогноз:")
    print(f"   Зміна ціни: {prediction['change_percent']:.3f}")
    print(f"   Впевненість: {prediction['confidence']:.3f}")
    print(f"   Індикатори: {len(prediction['indicators'])}")

    return avg_time


if __name__ == "__main__":
    benchmark_indicators()