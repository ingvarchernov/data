#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 АНАЛІЗ РИНКОВИХ УМОВ
Перевірка трендів, волатильності та визначення найкращих індикаторів
"""
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
import os
from dotenv import load_dotenv

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


def calculate_indicators(df):
    """Розрахувати технічні індикатори"""
    # SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean() if len(df) >= 200 else np.nan
    
    # EMA
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df


def analyze_trend(df):
    """Визначити поточний тренд"""
    current_price = df['close'].iloc[-1]
    sma_50 = df['sma_50'].iloc[-1]
    sma_200 = df['sma_200'].iloc[-1] if not pd.isna(df['sma_200'].iloc[-1]) else None
    
    # Нахил SMA
    sma_50_slope = (df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]) / 10
    
    trend = "NEUTRAL"
    strength = 0
    
    if sma_200 is not None:
        if current_price > sma_50 > sma_200:
            trend = "STRONG UPTREND"
            strength = 2
        elif current_price > sma_50:
            trend = "UPTREND"
            strength = 1
        elif current_price < sma_50 < sma_200:
            trend = "STRONG DOWNTREND"
            strength = -2
        elif current_price < sma_50:
            trend = "DOWNTREND"
            strength = -1
    else:
        if current_price > sma_50 and sma_50_slope > 0:
            trend = "UPTREND"
            strength = 1
        elif current_price < sma_50 and sma_50_slope < 0:
            trend = "DOWNTREND"
            strength = -1
    
    return {
        'trend': trend,
        'strength': strength,
        'current_price': current_price,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'distance_from_sma50': (current_price - sma_50) / sma_50 * 100
    }


def analyze_volatility(df):
    """Аналіз волатильності"""
    returns = df['close'].pct_change()
    
    # Стандартне відхилення (волатильність)
    volatility = returns.std() * 100
    
    # ATR як % від ціни
    atr = df['atr'].iloc[-1]
    atr_pct = atr / df['close'].iloc[-1] * 100
    
    # Bollinger Bands width
    bb_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1] * 100
    
    # Класифікація
    if volatility < 1.5:
        regime = "LOW"
    elif volatility < 3.0:
        regime = "MEDIUM"
    else:
        regime = "HIGH"
    
    return {
        'volatility': volatility,
        'atr_pct': atr_pct,
        'bb_width': bb_width,
        'regime': regime
    }


def test_indicators(df):
    """Тестувати які індикатори працюють краще"""
    results = {
        'rsi_oversold': 0,
        'rsi_overbought': 0,
        'bb_lower': 0,
        'bb_upper': 0,
        'macd_cross': 0,
        'sma_cross': 0
    }
    
    wins = {key: 0 for key in results.keys()}
    
    for i in range(50, len(df) - 5):  # Залишаємо 5 свічок для перевірки результату
        current = df.iloc[i]
        future_price = df['close'].iloc[i+5]
        current_price = current['close']
        
        # RSI Oversold
        if current['rsi'] < 30:
            results['rsi_oversold'] += 1
            if future_price > current_price:
                wins['rsi_oversold'] += 1
        
        # RSI Overbought
        if current['rsi'] > 70:
            results['rsi_overbought'] += 1
            if future_price < current_price:
                wins['rsi_overbought'] += 1
        
        # BB Lower
        if current_price < current['bb_lower']:
            results['bb_lower'] += 1
            if future_price > current_price:
                wins['bb_lower'] += 1
        
        # BB Upper
        if current_price > current['bb_upper']:
            results['bb_upper'] += 1
            if future_price < current_price:
                wins['bb_upper'] += 1
        
        # MACD Cross
        if i > 0:
            prev = df.iloc[i-1]
            if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                results['macd_cross'] += 1
                if future_price > current_price:
                    wins['macd_cross'] += 1
        
        # SMA Cross
        if i > 0:
            prev = df.iloc[i-1]
            if prev['close'] < prev['sma_50'] and current_price > current['sma_50']:
                results['sma_cross'] += 1
                if future_price > current_price:
                    wins['sma_cross'] += 1
    
    accuracy = {}
    for key in results.keys():
        if results[key] > 0:
            accuracy[key] = wins[key] / results[key] * 100
        else:
            accuracy[key] = 0
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Аналіз ринкових умов')
    parser.add_argument('--days', type=int, default=30, help='Кількість днів для аналізу')
    parser.add_argument('--symbols', type=str, default='ETHUSDT,BTCUSDT,BNBUSDT', 
                        help='Символи через кому')
    parser.add_argument('--interval', type=str, default='4h', help='Таймфрейм')
    
    args = parser.parse_args()
    symbols = args.symbols.split(',')
    
    print('='*100)
    print(f'📊 АНАЛІЗ РИНКОВИХ УМОВ')
    print(f'Період: {args.days} днів | Таймфрейм: {args.interval}')
    print('='*100)
    
    for symbol in symbols:
        print(f'\n🔸 {symbol}')
        print('-'*100)
        
        try:
            # Завантажити дані
            df = get_historical_data(symbol, args.days, args.interval)
            
            # Розрахувати індикатори
            df = calculate_indicators(df)
            
            # Аналіз тренду
            trend_info = analyze_trend(df)
            print(f"\n📈 ТРЕНД: {trend_info['trend']} (strength: {trend_info['strength']})")
            print(f"   Current Price: ${trend_info['current_price']:.2f}")
            print(f"   SMA 50: ${trend_info['sma_50']:.2f} ({trend_info['distance_from_sma50']:+.2f}%)")
            if trend_info['sma_200']:
                print(f"   SMA 200: ${trend_info['sma_200']:.2f}")
            
            # Аналіз волатильності
            vol_info = analyze_volatility(df)
            print(f"\n📊 ВОЛАТИЛЬНІСТЬ: {vol_info['regime']}")
            print(f"   Returns StdDev: {vol_info['volatility']:.2f}%")
            print(f"   ATR: {vol_info['atr_pct']:.2f}%")
            print(f"   BB Width: {vol_info['bb_width']:.2f}%")
            
            # Тестування індикаторів
            indicator_accuracy = test_indicators(df)
            print(f"\n🎯 ТОЧНІСТЬ ІНДИКАТОРІВ (next 5 candles):")
            for indicator, acc in sorted(indicator_accuracy.items(), key=lambda x: x[1], reverse=True):
                status = '✅' if acc >= 55 else '⚠️' if acc >= 45 else '❌'
                print(f"   {status} {indicator:20s}: {acc:5.1f}%")
            
            # Рекомендації
            print(f"\n💡 РЕКОМЕНДАЦІЇ:")
            if trend_info['strength'] > 0:
                print(f"   ✅ Тренд вгору - розглянути LONG позиції")
                print(f"   ⚠️ SHORT тільки при сильних сигналах reversal")
            elif trend_info['strength'] < 0:
                print(f"   ✅ Тренд вниз - розглянути SHORT позиції")
                print(f"   ⚠️ LONG тільки при сильних сигналах reversal")
            else:
                print(f"   ⚠️ Флет - використовувати mean reversion стратегії")
            
            if vol_info['regime'] == 'HIGH':
                print(f"   ⚠️ Висока волатильність - збільшити SL/TP")
            elif vol_info['regime'] == 'LOW':
                print(f"   ✅ Низька волатильність - зменшити SL/TP")
            
            best_indicator = max(indicator_accuracy.items(), key=lambda x: x[1])
            if best_indicator[1] >= 55:
                print(f"   ✅ Найкращий індикатор: {best_indicator[0]} ({best_indicator[1]:.1f}%)")
            
        except Exception as e:
            print(f"❌ Помилка: {e}")
    
    print('\n' + '='*100)
    print('✅ Аналіз завершено')
    print('='*100)


if __name__ == '__main__':
    main()
