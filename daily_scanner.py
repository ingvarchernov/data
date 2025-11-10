#!/usr/bin/env python3
"""
Daily Market Scanner - Multi-Timeframe Pattern Detection
Запускається щодня о 00:00 UTC
Сканує 150+ символів на 5 таймфреймах (1d, 4h, 1h, 30m, 15m)
"""
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import json

from config import TRADING_CONFIG
from multi_timeframe_scanner import MultiTimeframeScanner
from telegram_bot import telegram_notifier

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"daily_scan_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PatternDatabase:
    """Зберігає історію паттернів для виявлення повторень"""
    
    def __init__(self, db_file='pattern_history.json'):
        self.db_file = Path(db_file)
        self.patterns = self.load()
    
    def load(self):
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {'patterns': [], 'recurring': {}}
    
    def save(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def add_pattern(self, symbol, pattern_type, direction, confidence, timestamp, timeframe='15m'):
        """Додає паттерн до історії"""
        pattern_entry = {
            'symbol': symbol,
            'type': pattern_type,
            'direction': direction,
            'confidence': confidence,
            'timestamp': timestamp,
            'timeframe': timeframe
        }
        self.patterns['patterns'].append(pattern_entry)
        
        # Рахуємо повторення (з урахуванням таймфрейму)
        key = f"{symbol}_{pattern_type}_{direction}_{timeframe}"
        if key not in self.patterns['recurring']:
            self.patterns['recurring'][key] = {
                'count': 0,
                'last_seen': [],
                'avg_confidence': 0,
                'timeframe': timeframe
            }
        
        rec = self.patterns['recurring'][key]
        rec['count'] += 1
        rec['last_seen'].append(timestamp)
        rec['last_seen'] = rec['last_seen'][-10:]  # Keep last 10
        
        # Average confidence
        recent = [p for p in self.patterns['patterns'][-100:] 
                  if p['symbol'] == symbol and p['type'] == pattern_type and p['timeframe'] == timeframe]
        if recent:
            rec['avg_confidence'] = sum(p['confidence'] for p in recent) / len(recent)
        
        self.save()
        return rec['count']
    
    def get_recurring_patterns(self, min_count=3):
        """Повертає паттерни що повторюються"""
        recurring = []
        for key, data in self.patterns['recurring'].items():
            if data['count'] >= min_count:
                parts = key.split('_')
                if len(parts) >= 4:
                    symbol = parts[0]
                    ptype = '_'.join(parts[1:-2])  # Pattern type може містити _
                    direction = parts[-2]
                    timeframe = parts[-1]
                else:
                    # Old format compatibility
                    symbol, ptype, direction = key.split('_')
                    timeframe = '15m'
                
                recurring.append({
                    'symbol': symbol,
                    'type': ptype,
                    'direction': direction,
                    'timeframe': timeframe,
                    'count': data['count'],
                    'avg_confidence': data['avg_confidence'],
                    'last_seen': data['last_seen'][-1] if data['last_seen'] else None
                })
        
        # Sort by count
        recurring.sort(key=lambda x: x['count'], reverse=True)
        return recurring


async def daily_market_scan():
    """Щоденний мультитаймфреймовий скан ринку"""
    
    logger.info("="*100)
    logger.info("🔍 DAILY MULTI-TIMEFRAME MARKET SCAN")
    logger.info("="*100)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Timeframes: 1d, 4h, 1h, 30m, 15m")
    logger.info("="*100 + "\n")
    
    # Initialize MTF scanner
    scanner = MultiTimeframeScanner()
    pattern_db = PatternDatabase()
    
    # Get top 150 symbols by volume
    symbols = await scanner.get_top_symbols(limit=150)
    
    logger.info(f"📊 Scanning {len(symbols)} symbols on 5 timeframes...")
    logger.info(f"⚡ Total analysis points: {len(symbols) * 5} = {len(symbols)} symbols × 5 TFs\n")
    
    # Phase 1: Multi-timeframe scan
    start_time = datetime.now()
    results = await scanner.batch_scan(symbols, batch_size=20)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n✅ Scan completed in {elapsed:.1f}s ({len(symbols)/elapsed:.1f} symbols/sec)")
    logger.info(f"📈 Found patterns on {len(results)} symbols")
    
    # Phase 2: Store patterns and find recurring
    logger.info("\n" + "="*100)
    logger.info("📊 PATTERN ANALYSIS & RECURRING DETECTION")
    logger.info("="*100)
    
    timestamp = datetime.now().isoformat()
    new_patterns = []
    
    for result in results[:50]:  # Top 50
        symbol = result['symbol']
        
        # Extract all patterns from all timeframes
        for tf, patterns in result['timeframes'].items():
            for pattern in patterns:
                ptype = pattern.pattern_type.value
                direction = pattern.direction
                confidence = pattern.confidence
                
                # Add to database
                count = pattern_db.add_pattern(
                    symbol, ptype, direction, confidence, timestamp, timeframe=tf
                )
                
                new_patterns.append({
                    'symbol': symbol,
                    'type': ptype,
                    'direction': direction,
                    'confidence': confidence,
                    'timeframe': tf,
                    'count': count,
                    'mtf_score': result['mtf_score']
                })
    
    logger.info(f"✅ Stored {len(new_patterns)} new patterns")
    
    # Phase 3: Find high-probability recurring patterns
    logger.info("\n" + "="*100)
    logger.info("🎯 HIGH-PROBABILITY MULTI-TIMEFRAME SIGNALS")
    logger.info("="*100)
    
    recurring = pattern_db.get_recurring_patterns(min_count=3)
    
    if recurring:
        logger.info(f"Found {len(recurring)} recurring patterns:\n")
        
        trading_signals = []
        for rec in recurring[:20]:  # Top 20
            logger.info(
                f"  🔔 {rec['symbol']:12s} | {rec['type']:20s} | {rec['direction']:6s} | "
                f"Count: {rec['count']:3d}x | Confidence: {rec['avg_confidence']:.1f}%"
            )
            
            # High-probability signals (5+ occurrences, 65%+ confidence)
            if rec['count'] >= 5 and rec['avg_confidence'] >= 65.0:
                trading_signals.append(rec)
        
        # Add MTF confluence signals (patterns on multiple timeframes)
        logger.info("\n" + "="*100)
        logger.info("⚡ MULTI-TIMEFRAME CONFLUENCE SIGNALS")
        logger.info("="*100)
        
        for result in results[:10]:  # Top 10 MTF scores
            confluence = result['confluence']
            if confluence.get('same_direction', 0) >= 0.6:  # 60%+ same direction
                logger.info(
                    f"  ⭐ {result['symbol']:12s} | MTF Score: {result['mtf_score']:5.1f} | "
                    f"{confluence.get('dominant_direction', 'N/A'):6s} | "
                    f"Confluence: {confluence.get('same_direction', 0)*100:.0f}% | "
                    f"TFs: {len(confluence.get('timeframes_with_patterns', []))}/5"
                )
                
                # Add to signals if strong confluence
                if result['mtf_score'] >= 70 and len(confluence.get('timeframes_with_patterns', [])) >= 3:
                    trading_signals.append({
                        'symbol': result['symbol'],
                        'type': 'MTF_CONFLUENCE',
                        'direction': confluence.get('dominant_direction', 'N/A'),
                        'count': len(confluence.get('timeframes_with_patterns', [])),
                        'avg_confidence': result['mtf_score'],
                        'confluence': confluence.get('same_direction', 0) * 100
                    })
        
        # Send Telegram signals
        if trading_signals:
            logger.info(f"\n🚀 Sending {len(trading_signals)} trading signals to Telegram...")
            
            message = f"🚨 TRADING SIGNALS ALERT\n\n"
            message += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            message += f"Signals: {len(trading_signals)}\n\n"
            
            for sig in trading_signals[:5]:  # Top 5
                message += f"📊 {sig['symbol']}\n"
                message += f"  Pattern: {sig['type']}\n"
                message += f"  Direction: {sig['direction'].upper()}\n"
                
                if sig['type'] == 'MTF_CONFLUENCE':
                    message += f"  MTF Score: {sig['avg_confidence']:.1f}\n"
                    message += f"  Confluence: {sig.get('confluence', 0):.0f}%\n"
                    message += f"  Timeframes: {sig['count']}/5\n\n"
                else:
                    message += f"  Recurring: {sig['count']}x\n"
                    message += f"  Confidence: {sig['avg_confidence']:.1f}%\n\n"
            
            message += f"Total signals: {len(trading_signals)}\n"
            message += f"Log: {log_file.name}"
            
            await telegram_notifier.send_message(message)
            logger.info("✅ Signals sent!")
    else:
        logger.info("No recurring patterns found yet (need 3+ occurrences)")
    
    # Summary
    logger.info("\n" + "="*100)
    logger.info("📊 SUMMARY")
    logger.info("="*100)
    logger.info(f"Symbols scanned: {scanner.stats['symbols_scanned']}")
    logger.info(f"Timeframes analyzed: {scanner.stats['timeframes_analyzed']}")
    logger.info(f"Total patterns found: {scanner.stats['patterns_found']}")
    logger.info(f"New patterns stored: {len(new_patterns)}")
    logger.info(f"Recurring patterns: {len(recurring) if recurring else 0}")
    logger.info(f"Trading signals: {len(trading_signals) if 'trading_signals' in locals() else 0}")
    logger.info(f"Scan time: {elapsed:.1f}s")
    logger.info(f"Speed: {len(symbols)/elapsed:.1f} symbols/sec")
    logger.info("="*100)
    


async def scheduler():
    """Запускає скан кожні 24 години"""
    
    logger.info("� Multi-Timeframe Daily Scanner Started")
    logger.info("Schedule: Every day at 00:00 UTC")
    logger.info("Timeframes: 1d, 4h, 1h, 30m, 15m")
    logger.info("Symbols: ~150 (top by volume)")
    logger.info("Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Run scan
            await daily_market_scan()
            
            # Wait 24 hours
            logger.info(f"⏰ Next scan in 24 hours (at 00:00 UTC tomorrow)")
            await asyncio.sleep(24 * 60 * 60)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Scheduler stopped")
            break
        except Exception as e:
            logger.error(f"❌ Error in scheduler: {e}", exc_info=True)
            logger.info("Retrying in 1 hour...")
            await asyncio.sleep(3600)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Run once
        asyncio.run(daily_market_scan())
    else:
        # Run as scheduler
        asyncio.run(scheduler())
