#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 АНАЛІЗ НЕВДАЧ - Глибокий аналіз чому всі угоди збиткові
"""
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    database=os.getenv('DB_NAME', 'trading'),
    user=os.getenv('DB_USER', 'trader'),
    password=os.getenv('DB_PASSWORD', '')
)

today = datetime.now().date()

# Аналіз: які прогнози були правильні/неправильні
query = '''
SELECT 
    symbol,
    side,
    ml_prediction,
    ml_confidence,
    entry_price,
    exit_price,
    realized_pnl,
    realized_pnl_pct,
    exit_reason,
    entry_time,
    exit_time
FROM positions 
WHERE status = 'closed' 
  AND DATE(exit_time AT TIME ZONE 'UTC') = %s
ORDER BY symbol, exit_time
'''

cur = conn.cursor()
cur.execute(query, (today,))
rows = cur.fetchall()

print('='*120)
print('🔍 АНАЛІЗ ТОЧНОСТІ ПРОГНОЗІВ')
print('='*120)

by_symbol = {}
for row in rows:
    symbol, side, prediction, confidence, entry, exit_price, pnl, pnl_pct, reason, entry_time, exit_time = row
    
    if symbol not in by_symbol:
        by_symbol[symbol] = []
    
    # Чи був прогноз правильним?
    entry_f = float(entry)
    exit_f = float(exit_price)
    
    if side == 'LONG':
        actual_direction = 'UP' if exit_f > entry_f else 'DOWN'
    else:  # SHORT
        actual_direction = 'DOWN' if exit_f > entry_f else 'UP'
    
    correct = (prediction == actual_direction)
    
    # Час утримування
    duration = (exit_time - entry_time).total_seconds() / 60  # хвилини
    
    by_symbol[symbol].append({
        'side': side,
        'prediction': prediction,
        'confidence': float(confidence) if confidence else 0,
        'actual': actual_direction,
        'correct': correct,
        'pnl': float(pnl) if pnl else 0,
        'pnl_pct': float(pnl_pct) if pnl_pct else 0,
        'duration': duration
    })

print(f"\n{'Symbol':<12} {'Trades':>7} {'Correct':>8} {'Accuracy':>9} {'Avg Conf':>9} {'Avg PnL':>10} {'Avg Time':>10}")
print('-'*120)

total_trades = 0
total_correct = 0
total_pnl = 0

for symbol in sorted(by_symbol.keys()):
    trades = by_symbol[symbol]
    correct = sum(1 for t in trades if t['correct'])
    accuracy = correct / len(trades) * 100 if trades else 0
    avg_conf = sum(t['confidence'] for t in trades) / len(trades) if trades else 0
    avg_pnl = sum(t['pnl'] for t in trades) / len(trades) if trades else 0
    avg_time = sum(t['duration'] for t in trades) / len(trades) if trades else 0
    
    total_trades += len(trades)
    total_correct += correct
    total_pnl += sum(t['pnl'] for t in trades)
    
    status = '✅' if accuracy >= 50 else '❌'
    print(f"{status} {symbol:<12} {len(trades):>7d} {correct:>8d} {accuracy:>8.1f}% {avg_conf:>8.1f}% ${avg_pnl:>9.2f} {avg_time:>8.1f}m")

print('-'*120)
overall_accuracy = total_correct / total_trades * 100 if total_trades > 0 else 0
print(f'ВСЬОГО: {total_trades} угод | ML Accuracy: {overall_accuracy:.1f}% | Total PnL: ${total_pnl:.2f}')

print('\n'+'='*120)
print('📊 ДЕТАЛІ ПО СИМВОЛАМ')
print('='*120)
for symbol in sorted(by_symbol.keys()):
    print(f'\n🔸 {symbol}:')
    for i, t in enumerate(by_symbol[symbol], 1):
        status = '✅' if t['correct'] else '❌'
        print(f"  {status} #{i}: Predicted {t['prediction']} ({t['confidence']:.1f}%), Actual: {t['actual']}, "
              f"{t['side']}, PnL: ${t['pnl']:.2f} ({t['pnl_pct']:.1f}%), Duration: {t['duration']:.1f}m")

# Додатковий аналіз: чому SHORT ETHUSDT не працює?
print('\n'+'='*120)
print('🎯 СПЕЦІАЛЬНИЙ АНАЛІЗ: ETHUSDT SHORT')
print('='*120)

if 'ETHUSDT' in by_symbol:
    ethusdt_trades = [t for t in by_symbol['ETHUSDT'] if t['side'] == 'SHORT']
    if ethusdt_trades:
        print(f"Всього SHORT угод: {len(ethusdt_trades)}")
        print(f"Правильних прогнозів: {sum(1 for t in ethusdt_trades if t['correct'])}")
        print(f"Середня впевненість: {sum(t['confidence'] for t in ethusdt_trades)/len(ethusdt_trades):.1f}%")
        print(f"Середній PnL: ${sum(t['pnl'] for t in ethusdt_trades)/len(ethusdt_trades):.2f}")
        print("\n⚠️ ПРОБЛЕМА: Модель передбачає DOWN з високою впевненістю, але ціна йде UP")
        print("   Можливі причини:")
        print("   1. Модель натренована на старих даних (trend reversal)")
        print("   2. Волатильність вища ніж очікувалося")
        print("   3. Неправильна інтерпретація індикаторів")

# Аналіз волатильності
print('\n'+'='*120)
print('📈 АНАЛІЗ ВОЛАТИЛЬНОСТІ')
print('='*120)

query_volatility = '''
SELECT 
    symbol,
    entry_price,
    exit_price,
    ABS((exit_price - entry_price) / entry_price * 100) as price_move_pct
FROM positions 
WHERE status = 'closed' 
  AND DATE(exit_time AT TIME ZONE 'UTC') = %s
'''

cur.execute(query_volatility, (today,))
vol_rows = cur.fetchall()

vol_by_symbol = {}
for symbol, entry, exit_price, move_pct in vol_rows:
    if symbol not in vol_by_symbol:
        vol_by_symbol[symbol] = []
    vol_by_symbol[symbol].append(float(move_pct))

print(f"\n{'Symbol':<12} {'Avg Move':>10} {'Max Move':>10} {'Min Move':>10}")
print('-'*120)

for symbol in sorted(vol_by_symbol.keys()):
    moves = vol_by_symbol[symbol]
    avg_move = sum(moves) / len(moves)
    max_move = max(moves)
    min_move = min(moves)
    
    status = '⚠️' if avg_move > 5 else '✅'
    print(f"{status} {symbol:<12} {avg_move:>9.2f}% {max_move:>9.2f}% {min_move:>9.2f}%")

print('\n'+'='*120)
print('💡 РЕКОМЕНДАЦІЇ')
print('='*120)
print("""
1. ❌ ML МОДЕЛІ НЕ ПРАЦЮЮТЬ (0% accuracy означає що прогнози систематично неправильні)
   → Потрібно ПОВНІСТЮ перетренувати на СВІЖИХ даних (останні 30-60 днів)
   
2. ⚠️ ETHUSDT SHORT - найбільша проблема
   → Модель каже DOWN (75-88% confidence), але ціна йде UP
   → Можливо, ринок в тренді вгору, а модель натренована на флеті
   
3. 🔄 АЛЬТЕРНАТИВНІ СТРАТЕГІЇ:
   a) Mean Reversion - торгівля на відскоках від середніх
   b) Momentum - торгівля в напрямку тренду (не проти)
   c) Volatility Breakout - торгівля на пробоях волатильності
   d) Support/Resistance - торгівля від рівнів
   
4. 📊 ТЕСТУВАННЯ:
   → Створити backtesting систему для перевірки стратегій
   → Протестувати на історичних даних за останні 30 днів
   → Порівняти результати різних підходів
   
5. 🛡️ ЗАХИСТ:
   → SL 1.5% занадто широкий для force close при -7%
   → Розглянути adaptive SL/TP залежно від волатильності
   → Додати trailing stop для збереження прибутку
""")

cur.close()
conn.close()

print('='*120)
print('✅ Аналіз завершено')
print('='*120)
