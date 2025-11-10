#!/usr/bin/env python3
"""
Multi-Timeframe Pattern Scanner - Швидкий скан 100+ символів на 5 таймфреймах
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import json
from collections import defaultdict

from config import TRADING_CONFIG
from pattern_scanner import PatternScanner
from volatility_filter import VolatilityFilter
from unified_binance_loader import UnifiedBinanceLoader
from telegram_bot import telegram_notifier

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"mtf_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Multi-timeframe setup
TIMEFRAMES = ['1d', '4h', '1h', '30m', '15m']

# Timeframe weights (higher = more important)
TF_WEIGHTS = {
    '1d': 5.0,   # Найважливіший
    '4h': 3.0,
    '1h': 2.0,
    '30m': 1.5,
    '15m': 1.0
}

# Days back for each timeframe
TF_DAYS_BACK = {
    '1d': 90,   # 3 місяці
    '4h': 30,   # 1 місяць
    '1h': 14,   # 2 тижні
    '30m': 7,   # 1 тиждень
    '15m': 7    # 1 тиждень
}

# Batch settings for speed
BATCH_SIZE = 20  # Скільки символів завантажувати паралельно
MAX_CONCURRENT = 50  # Максимум паралельних запитів


class MultiTimeframeScanner:
    """Швидкий мультитаймфреймовий сканер"""
    
    def __init__(self):
        self.loader = UnifiedBinanceLoader(testnet=False)
        self.pattern_scanner = PatternScanner()
        self.volatility_filter = VolatilityFilter(min_score=25.0)  # Нижчий поріг для більшого покриття
        
        # Cache для швидкості
        self.data_cache = {}
        self.pattern_cache = {}
        
        # Statistics
        self.stats = {
            'symbols_scanned': 0,
            'timeframes_analyzed': 0,
            'patterns_found': 0,
            'api_calls': 0,
            'cache_hits': 0
        }
    
    async def get_top_symbols(self, limit: int = 150) -> List[str]:
        """Отримує топ символів з Binance за об'ємом торгів"""
        logger.info(f"📊 Fetching top {limit} symbols from Binance...")
        
        try:
            # Отримуємо тікери з Binance
            from binance.client import Client
            client = Client()
            
            # Отримуємо всі USDT пари
            tickers = client.get_ticker()
            
            # Фільтруємо USDT пари та сортуємо за об'ємом
            usdt_pairs = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT') and not symbol.endswith('DOWNUSDT') and not symbol.endswith('UPUSDT'):
                    try:
                        volume_usd = float(ticker['quoteVolume'])
                        usdt_pairs.append({
                            'symbol': symbol,
                            'volume': volume_usd,
                            'price_change': float(ticker['priceChangePercent'])
                        })
                    except (ValueError, KeyError):
                        continue
            
            # Сортуємо за об'ємом
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            
            # Беремо топ
            top_symbols = [p['symbol'] for p in usdt_pairs[:limit]]
            
            logger.info(f"✅ Found {len(top_symbols)} symbols")
            logger.info(f"Top 10: {', '.join(top_symbols[:10])}")
            
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            # Fallback to default list
            return self._get_default_symbols()
    
    def _get_default_symbols(self) -> List[str]:
        """Fallback список символів якщо API недоступний"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT',
            'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'ETCUSDT', 'TRXUSDT',
            'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT',
            'MATICUSDT', 'SHIBUSDT', 'FILUSDT', 'ICPUSDT', 'LDOUSDT',
            'SUIUSDT', 'PEPEUSDT', 'WIFUSDT', 'TAOUSDT', 'RNDRUSDT',
            'AAVEUSDT', 'FTMUSDT', 'GALAUSDT', 'SANDUSDT', 'MANAUSDT',
            'AXSUSDT', 'THETAUSDT', 'XLMUSDT', 'ALGOUSDT', 'VETUSDT',
            'GRTUSDT', 'MKRUSDT', 'COMPUSDT', 'SNXUSDT', 'ARUSDT',
            'RUNEUSDT', 'EGLDUSDT', 'ZILUSDT', 'WAVESUSDT', 'CRVUSDT'
        ]
    
    async def scan_symbol_mtf(
        self, 
        symbol: str, 
        timeframes: List[str] = None
    ) -> Dict[str, List]:
        """Сканує один символ на всіх таймфреймах паралельно"""
        if timeframes is None:
            timeframes = TIMEFRAMES
        
        results = {}
        
        # Створюємо задачі для паралельного виконання
        tasks = []
        for tf in timeframes:
            task = self._scan_single_tf(symbol, tf)
            tasks.append((tf, task))
        
        # Виконуємо всі таймфрейми паралельно
        for tf, task in tasks:
            try:
                patterns = await task
                results[tf] = patterns
                self.stats['timeframes_analyzed'] += 1
                if patterns:
                    self.stats['patterns_found'] += len(patterns)
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {tf}: {e}")
                results[tf] = []
        
        return results
    
    async def _scan_single_tf(self, symbol: str, timeframe: str) -> List:
        """Сканує один символ на одному таймфреймі"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Перевірка кешу
        if cache_key in self.pattern_cache:
            self.stats['cache_hits'] += 1
            return self.pattern_cache[cache_key]
        
        try:
            # Завантажуємо дані
            days_back = TF_DAYS_BACK.get(timeframe, 7)
            
            df = await self.loader.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                days_back=days_back
            )
            
            self.stats['api_calls'] += 1
            
            if df is None or len(df) < 50:
                return []
            
            # Швидка перевірка волатільності (тільки для вищих ТФ)
            if timeframe in ['1d', '4h', '1h']:
                is_tradeable, score, _ = self.volatility_filter.is_tradeable(df)
                if not is_tradeable:
                    return []
            
            # Детекція паттернів (використовуємо існуючий PatternScanner)
            df = df.reset_index()
            df = self.pattern_scanner._calculate_indicators(df)
            
            patterns = []
            
            # Rust patterns (швидко!)
            if self.pattern_scanner.use_rust and self.pattern_scanner.rust_detector:
                rust_patterns = self.pattern_scanner._detect_rust_patterns(df, symbol)
                patterns.extend(rust_patterns)
            
            # Python patterns
            patterns.extend(self.pattern_scanner._detect_chart_patterns(df, symbol))
            
            # Фільтруємо (знижений поріг для більшого покриття)
            filtered = []
            for p in patterns:
                if p.confidence >= 55:  # Нижче ніж стандартні 60
                    # Додаємо інфо про таймфрейм
                    p.details['timeframe'] = timeframe
                    p.details['tf_weight'] = TF_WEIGHTS.get(timeframe, 1.0)
                    filtered.append(p)
            
            # Кешуємо результат
            self.pattern_cache[cache_key] = filtered[:5]  # Топ 5 на ТФ
            
            return filtered[:5]
            
        except Exception as e:
            logger.debug(f"Error scanning {symbol} {timeframe}: {e}")
            return []
    
    def calculate_mtf_score(self, mtf_results: Dict[str, List]) -> float:
        """Розраховує загальний скор на основі всіх таймфреймів"""
        total_score = 0.0
        total_weight = 0.0
        
        for tf, patterns in mtf_results.items():
            if not patterns:
                continue
            
            tf_weight = TF_WEIGHTS.get(tf, 1.0)
            
            # Беремо найкращий паттерн з таймфрейму
            best_confidence = max(p.confidence for p in patterns)
            
            total_score += best_confidence * tf_weight
            total_weight += tf_weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def analyze_mtf_confluence(self, mtf_results: Dict[str, List]) -> Dict:
        """Аналізує confluence (співпадіння) паттернів на різних ТФ"""
        confluence = {
            'same_direction': 0,
            'pattern_types': defaultdict(int),
            'directions': defaultdict(int),
            'timeframes_with_patterns': []
        }
        
        for tf, patterns in mtf_results.items():
            if not patterns:
                continue
            
            confluence['timeframes_with_patterns'].append(tf)
            
            for pattern in patterns:
                confluence['pattern_types'][pattern.pattern_type.value] += 1
                confluence['directions'][pattern.direction] += 1
        
        # Рахуємо чи більшість паттернів в одному напрямку
        if confluence['directions']:
            max_dir = max(confluence['directions'].items(), key=lambda x: x[1])
            total_patterns = sum(confluence['directions'].values())
            confluence['same_direction'] = max_dir[1] / total_patterns if total_patterns > 0 else 0
            confluence['dominant_direction'] = max_dir[0]
        
        return dict(confluence)
    
    async def batch_scan(
        self, 
        symbols: List[str], 
        batch_size: int = BATCH_SIZE
    ) -> List[Dict]:
        """Сканує символи пакетами для максимальної швидкості"""
        logger.info(f"🚀 Starting batch scan: {len(symbols)} symbols, batch_size={batch_size}")
        
        all_results = []
        
        # Розбиваємо на батчі
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} symbols")
            
            # Скануємо батч паралельно
            tasks = [self.scan_symbol_mtf(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обробляємо результати
            for symbol, mtf_result in zip(batch, batch_results):
                if isinstance(mtf_result, Exception):
                    logger.warning(f"⚠️ {symbol}: {mtf_result}")
                    continue
                
                # Рахуємо скор та confluence
                mtf_score = self.calculate_mtf_score(mtf_result)
                confluence = self.analyze_mtf_confluence(mtf_result)
                
                # Підраховуємо загальну кількість паттернів
                total_patterns = sum(len(patterns) for patterns in mtf_result.values())
                
                if total_patterns > 0:
                    all_results.append({
                        'symbol': symbol,
                        'mtf_score': mtf_score,
                        'total_patterns': total_patterns,
                        'confluence': confluence,
                        'timeframes': mtf_result
                    })
                    
                    self.stats['symbols_scanned'] += 1
            
            # Невелика пауза між батчами
            await asyncio.sleep(0.1)
        
        # Сортуємо за MTF score
        all_results.sort(key=lambda x: x['mtf_score'], reverse=True)
        
        return all_results
    
    def print_results(self, results: List[Dict], top_n: int = 30):
        """Виводить результати"""
        logger.info("\n" + "="*100)
        logger.info(f"📊 TOP {top_n} MULTI-TIMEFRAME SIGNALS")
        logger.info("="*100)
        
        for i, result in enumerate(results[:top_n], 1):
            symbol = result['symbol']
            score = result['mtf_score']
            total_patterns = result['total_patterns']
            confluence = result['confluence']
            
            # Символ та скор
            logger.info(f"\n{i:2d}. {symbol:12s} | MTF Score: {score:5.1f} | Patterns: {total_patterns:3d}")
            
            # Confluence інфо
            same_dir = confluence.get('same_direction', 0) * 100
            dominant_dir = confluence.get('dominant_direction', 'N/A')
            tf_count = len(confluence.get('timeframes_with_patterns', []))
            
            logger.info(f"    Timeframes: {tf_count}/5 | Direction: {dominant_dir} ({same_dir:.0f}% confluence)")
            
            # Топ паттерни з кожного таймфрейму
            for tf in TIMEFRAMES:
                patterns = result['timeframes'].get(tf, [])
                if patterns:
                    best = patterns[0]
                    logger.info(
                        f"    [{tf:3s}] {best.pattern_type.value:20s} "
                        f"{best.direction:6s} - {best.confidence:.0f}%"
                    )
        
        logger.info("\n" + "="*100)
        logger.info("📈 STATISTICS")
        logger.info("="*100)
        logger.info(f"Symbols scanned: {self.stats['symbols_scanned']}")
        logger.info(f"Timeframes analyzed: {self.stats['timeframes_analyzed']}")
        logger.info(f"Total patterns found: {self.stats['patterns_found']}")
        logger.info(f"API calls: {self.stats['api_calls']}")
        logger.info(f"Cache hits: {self.stats['cache_hits']}")
        logger.info(f"Cache efficiency: {self.stats['cache_hits'] / max(self.stats['api_calls'], 1) * 100:.1f}%")
        logger.info("="*100)


async def main():
    """Головна функція"""
    logger.info("="*100)
    logger.info("🚀 MULTI-TIMEFRAME PATTERN SCANNER")
    logger.info("="*100)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Timeframes: {', '.join(TIMEFRAMES)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("="*100 + "\n")
    
    scanner = MultiTimeframeScanner()
    
    # Отримуємо топ символи
    symbols = await scanner.get_top_symbols(limit=150)
    
    # Сканування
    start_time = datetime.now()
    results = await scanner.batch_scan(symbols, batch_size=BATCH_SIZE)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Виводимо результати
    scanner.print_results(results, top_n=30)
    
    logger.info(f"\n⏱️  Total time: {elapsed:.1f}s")
    logger.info(f"📊 Speed: {len(symbols) / elapsed:.1f} symbols/sec")
    logger.info(f"📁 Log saved to: {log_file}\n")
    
    # Зберігаємо топ результати
    output_file = Path("mtf_signals.json")
    with open(output_file, 'w') as f:
        json.dump(results[:50], f, indent=2, default=str)
    
    logger.info(f"💾 Top 50 results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
