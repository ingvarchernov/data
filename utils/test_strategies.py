#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 ТЕСТУВАННЯ СТРАТЕГІЙ
Швидка перевірка чи генеруються сигнали
"""
import sys
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv

# Додати шлях до strategies
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy

load_dotenv()

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)


def get_historical_data(symbol, days=30, interval='4h'):
    """Завантажити історичні дані"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000),
        limit=1000
    )
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df


def test_strategy(strategy, symbol, df):
    """Тестувати стратегію"""
    print(f"\n{'='*80}")
    print(f"📊 {strategy.name} - {symbol}")
    print('='*80)
    
    signal = strategy.generate_signal(df, symbol)
    
    if signal:
        print(f"✅ СИГНАЛ ЗГЕНЕРОВАНО!")
        print(f"   Direction: {signal.direction}")
        print(f"   Confidence: {signal.confidence:.1f}%")
        print(f"   Entry: ${signal.entry_price:.4f}")
        print(f"   Stop Loss: ${signal.stop_loss:.4f} ({(signal.stop_loss/signal.entry_price-1)*100:.2f}%)")
        print(f"   Take Profit: ${signal.take_profit:.4f} ({(signal.take_profit/signal.entry_price-1)*100:.2f}%)")
        print(f"   Reason: {signal.reason}")
        
        # Валідація
        if strategy.validate_signal(signal):
            print(f"   ✅ Сигнал валідний")
        else:
            print(f"   ❌ Сигнал НЕ валідний")
        
        # Position size
        account_balance = 5000
        pos_size = strategy.calculate_position_size(signal, account_balance)
        print(f"   Position Size: ${pos_size:.2f} (Balance: ${account_balance})")
        
    else:
        print(f"❌ Немає сигналу")


def main():
    print('='*80)
    print('🧪 ТЕСТУВАННЯ ТОРГОВИХ СТРАТЕГІЙ')
    print('='*80)
    
    # Тестові символи (проблемні пари)
    symbols = ['ETHUSDT', 'ATOMUSDT', 'AVAXUSDT', 'VETUSDT']
    
    # Створення стратегій
    trend_strategy = TrendFollowingStrategy()
    mean_reversion_strategy = MeanReversionStrategy()
    
    for symbol in symbols:
        print(f"\n\n{'🔸 ' + symbol:=^80}")
        
        try:
            # Завантажити дані
            df = get_historical_data(symbol, days=60, interval='4h')
            print(f"✅ Завантажено {len(df)} свічок")
            
            # Тестувати обидві стратегії
            test_strategy(trend_strategy, symbol, df)
            test_strategy(mean_reversion_strategy, symbol, df)
            
        except Exception as e:
            print(f"❌ Помилка: {e}")
            import traceback
            traceback.print_exc()
    
    print('\n' + '='*80)
    print('✅ Тестування завершено')
    print('='*80)
    print('\n💡 ВИСНОВКИ:')
    print('1. Якщо Trend Following генерує SHORT сигнали - добре (ринок в downtrend)')
    print('2. Якщо Mean Reversion генерує сигнали - можна використати для контр-тренду')
    print('3. Перевірте чи валідні сигнали та чи правильні SL/TP')
    print('4. Наступний крок: backtesting на історичних даних\n')


if __name__ == '__main__':
    main()
