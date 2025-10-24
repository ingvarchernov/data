#!/usr/bin/env python3
"""
Тестування різних стратегій інтерпретації сигналів моделі
"""
import pandas as pd
import numpy as np
import sys

def test_strategy(df: pd.DataFrame, invert: bool = False, name: str = "Original"):
    """Тестування стратегії"""
    
    # Розрахунок руху ціни
    df = df.copy()
    df['price_change'] = df['price'].shift(-1) - df['price']
    df['price_direction'] = np.where(df['price_change'] > 0, 'UP', 
                            np.where(df['price_change'] < 0, 'DOWN', 'FLAT'))
    
    # Видалимо останній рядок
    df = df[:-1]
    
    # Інверсія сигналів якщо потрібно
    if invert:
        df['prediction'] = df['prediction'].replace({'UP': 'DOWN', 'DOWN': 'UP'})
    
    # Статистика
    correct_up = ((df['prediction'] == 'UP') & (df['price_direction'] == 'UP')).sum()
    wrong_up = ((df['prediction'] == 'UP') & (df['price_direction'] == 'DOWN')).sum()
    
    correct_down = ((df['prediction'] == 'DOWN') & (df['price_direction'] == 'DOWN')).sum()
    wrong_down = ((df['prediction'] == 'DOWN') & (df['price_direction'] == 'UP')).sum()
    
    correct_neutral = ((df['prediction'] == 'NEUTRAL') & (df['price_direction'] == 'FLAT')).sum()
    
    total_up = (df['prediction'] == 'UP').sum()
    total_down = (df['prediction'] == 'DOWN').sum()
    total_neutral = (df['prediction'] == 'NEUTRAL').sum()
    
    total_correct = correct_up + correct_down + correct_neutral
    total_signals = len(df)
    
    # Розрахунок прибутковості
    df['signal_profit'] = 0.0
    df.loc[df['prediction'] == 'UP', 'signal_profit'] = df['price_change']
    df.loc[df['prediction'] == 'DOWN', 'signal_profit'] = -df['price_change']
    
    total_profit = df['signal_profit'].sum()
    avg_profit_per_trade = df[df['prediction'] != 'NEUTRAL']['signal_profit'].mean()
    
    print(f"\n{'='*80}")
    print(f"📊 СТРАТЕГІЯ: {name}")
    print(f"{'='*80}")
    
    if total_up > 0:
        up_accuracy = correct_up / total_up * 100
        print(f"📈 UP сигналів: {total_up}")
        print(f"   ✅ Правильно: {correct_up} ({up_accuracy:.1f}%)")
        print(f"   ❌ Помилково: {wrong_up} ({100-up_accuracy:.1f}%)")
    else:
        print(f"📈 UP сигналів: 0")
    
    if total_down > 0:
        down_accuracy = correct_down / total_down * 100
        print(f"📉 DOWN сигналів: {total_down}")
        print(f"   ✅ Правильно: {correct_down} ({down_accuracy:.1f}%)")
        print(f"   ❌ Помилково: {wrong_down} ({100-down_accuracy:.1f}%)")
    else:
        print(f"📉 DOWN сигналів: 0")
    
    print(f"⚪ NEUTRAL сигналів: {total_neutral}")
    
    overall_accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
    print(f"\n🎯 Загальна точність: {overall_accuracy:.1f}%")
    print(f"💰 Сумарний прибуток: ${total_profit:,.2f}")
    print(f"📊 Середній прибуток на сигнал: ${avg_profit_per_trade:,.2f}")
    
    return {
        'name': name,
        'accuracy': overall_accuracy,
        'up_accuracy': correct_up / total_up * 100 if total_up > 0 else 0,
        'down_accuracy': correct_down / total_down * 100 if total_down > 0 else 0,
        'total_profit': total_profit,
        'avg_profit': avg_profit_per_trade
    }


def main():
    csv_file = 'graphics/csv/classification_analysis_BTCUSDT_20251023_224625.csv'
    
    print("🔍 ТЕСТУВАННЯ СТРАТЕГІЙ ІНТЕРПРЕТАЦІЇ СИГНАЛІВ")
    print("="*80)
    
    df = pd.read_csv(csv_file)
    
    results = []
    
    # 1. Оригінальна стратегія
    results.append(test_strategy(df, invert=False, name="Оригінальна (як є)"))
    
    # 2. Інвертована стратегія
    results.append(test_strategy(df, invert=True, name="Інвертована (UP↔DOWN)"))
    
    # Порівняння
    print(f"\n{'='*80}")
    print("📊 ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
    print(f"{'='*80}")
    
    best = max(results, key=lambda x: x['accuracy'])
    
    for r in results:
        marker = "🏆" if r == best else "  "
        print(f"{marker} {r['name']:25s} | Точність: {r['accuracy']:5.1f}% | Прибуток: ${r['total_profit']:+8.2f}")
    
    print(f"\n✅ РЕКОМЕНДАЦІЯ: {'Інвертувати сигнали!' if best['name'].startswith('Інвертована') else 'Використовувати як є'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
