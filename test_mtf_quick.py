#!/usr/bin/env python3
"""Quick MTF test - 30 symbols, all timeframes"""
import asyncio
from multi_timeframe_scanner import MultiTimeframeScanner

async def quick_test():
    print("="*80)
    print("🚀 QUICK MTF TEST - 30 symbols × 5 timeframes")
    print("="*80)
    
    scanner = MultiTimeframeScanner()
    
    # Top 30 symbols
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT',
        'LTCUSDT', 'DOTUSDT', 'TRXUSDT', 'NEARUSDT', 'ICPUSDT',
        'FILUSDT', 'SUIUSDT', 'PEPEUSDT', 'WIFUSDT', 'TAOUSDT',
        'ARBUSDT', 'OPUSDT', 'AAVEUSDT', 'ATOMUSDT', 'INJUSDT',
        'HBARUSDT', 'FETUSDT', 'VIRTUALUSDT', 'TRUMPUSDT', 'STRKUSDT'
    ]
    
    print(f"📊 Symbols: {len(symbols)}")
    print(f"📈 Total analysis points: {len(symbols) * 5}\n")
    
    from datetime import datetime
    start = datetime.now()
    
    # Scan
    results = await scanner.batch_scan(symbols, batch_size=15)
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # Print top 10
    print(f"\n{'='*80}")
    print(f"📊 TOP 10 MTF SIGNALS")
    print(f"{'='*80}\n")
    
    for i, r in enumerate(results[:10], 1):
        conf = r['confluence']
        tf_count = len(conf.get('timeframes_with_patterns', []))
        direction = conf.get('dominant_direction', 'N/A')
        confluence_pct = conf.get('same_direction', 0) * 100
        
        print(f"{i:2d}. {r['symbol']:12s} | Score: {r['mtf_score']:5.1f} | "
              f"TFs: {tf_count}/5 | {direction:6s} | Confluence: {confluence_pct:3.0f}%")
    
    print(f"\n{'='*80}")
    print(f"⏱️  Time: {elapsed:.1f}s | Speed: {len(symbols)/elapsed:.1f} symbols/sec")
    print(f"📊 Patterns: {scanner.stats['patterns_found']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(quick_test())
