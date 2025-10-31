#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 VOLATILITY SCANNER
Сканер волатильності для пошуку найактивніших монет
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VolatilityScanner:
    """Сканер для пошуку волатильних монет"""
    
    def __init__(self, client, symbols: List[str]):
        self.client = client
        self.all_symbols = symbols
        self.volatility_cache = {}
        self.cache_time = {}
        self.cache_duration = 900  # 15 хвилин
    
    async def get_24h_stats(self, symbol: str) -> Dict:
        """Отримання 24-годинної статистики"""
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_ticker(symbol=symbol)
            )
            return {
                'symbol': symbol,
                'price_change_pct': float(ticker['priceChangePercent']),
                'volume': float(ticker['quoteVolume']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice']),
                'trades': int(ticker['count'])
            }
        except Exception as e:
            logger.error(f"❌ Помилка статистики {symbol}: {e}")
            return None
    
    async def calculate_volatility_score(self, symbol: str) -> float:
        """
        Розрахунок скору волатильності (0-100)
        Враховує:
        - Ціновий діапазон (high-low)
        - Об'єм торгів
        - Кількість трейдів
        - Зміну ціни за 24h
        """
        try:
            stats = await self.get_24h_stats(symbol)
            if not stats:
                return 0.0
            
            # 1. Ціновий діапазон (чим більший - тим краще)
            price_range_pct = ((stats['high'] - stats['low']) / stats['low']) * 100
            range_score = min(price_range_pct * 2, 40)  # Максимум 40 балів
            
            # 2. Зміна ціни за 24h (абсолютне значення)
            change_score = min(abs(stats['price_change_pct']) * 2, 30)  # Максимум 30 балів
            
            # 3. Об'єм (нормалізований)
            # Мінімум $10M для високого скору
            volume_score = min((stats['volume'] / 10_000_000) * 20, 20)  # Максимум 20 балів
            
            # 4. Кількість трейдів (активність)
            # Мінімум 50k трейдів для високого скору
            trades_score = min((stats['trades'] / 50_000) * 10, 10)  # Максимум 10 балів
            
            total_score = range_score + change_score + volume_score + trades_score
            
            return round(total_score, 2)
            
        except Exception as e:
            logger.error(f"❌ Помилка розрахунку волатильності {symbol}: {e}")
            return 0.0
    
    async def scan_all_symbols(self) -> List[Tuple[str, float, Dict]]:
        """
        Сканування всіх символів з розрахунком скору
        Повертає: [(symbol, score, stats), ...]
        """
        logger.info(f"🔍 Сканування волатильності {len(self.all_symbols)} монет...")
        
        results = []
        tasks = []
        
        # Паралельний збір статистики
        for symbol in self.all_symbols:
            tasks.append(self.get_24h_stats(symbol))
        
        stats_list = await asyncio.gather(*tasks)
        
        # Розрахунок скорів
        for stats in stats_list:
            if stats:
                symbol = stats['symbol']
                
                # Ціновий діапазон
                price_range_pct = ((stats['high'] - stats['low']) / stats['low']) * 100
                range_score = min(price_range_pct * 2, 40)
                
                # Зміна ціни
                change_score = min(abs(stats['price_change_pct']) * 2, 30)
                
                # Об'єм
                volume_score = min((stats['volume'] / 10_000_000) * 20, 20)
                
                # Трейди
                trades_score = min((stats['trades'] / 50_000) * 10, 10)
                
                total_score = range_score + change_score + volume_score + trades_score
                
                results.append((symbol, round(total_score, 2), stats))
        
        # Сортування за скором (найкращі спочатку)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def get_top_volatile_symbols(self, count: int = 6, min_score: float = 30.0) -> List[str]:
        """
        Отримання топ N найволатильніших символів
        
        Args:
            count: Кількість символів
            min_score: Мінімальний скор (30-50 - помірна, 50+ - висока)
        
        Returns:
            Список символів
        """
        # Перевірка кешу
        now = datetime.now()
        if 'symbols' in self.volatility_cache:
            cache_age = (now - self.cache_time.get('symbols', datetime.min)).total_seconds()
            if cache_age < self.cache_duration:
                logger.info(f"📦 Використання кешу волатильності (вік: {int(cache_age)}s)")
                return self.volatility_cache['symbols'][:count]
        
        # Нове сканування
        results = await self.scan_all_symbols()
        
        # Фільтрація за мінімальним скором
        filtered = [(s, sc, st) for s, sc, st in results if sc >= min_score]
        
        if not filtered:
            logger.warning(f"⚠️ Не знайдено монет зі скором >= {min_score}, знижуємо до {min_score * 0.7:.1f}")
            filtered = [(s, sc, st) for s, sc, st in results if sc >= min_score * 0.7]
        
        # Логування результатів
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 ВОЛАТИЛЬНІСТЬ (топ {min(10, len(filtered))})")
        logger.info(f"{'='*70}")
        for i, (symbol, score, stats) in enumerate(filtered[:10], 1):
            logger.info(
                f"{i:2d}. {symbol:12s} | "
                f"Score: {score:5.1f} | "
                f"Range: {((stats['high']-stats['low'])/stats['low']*100):5.2f}% | "
                f"Change: {stats['price_change_pct']:+6.2f}% | "
                f"Vol: ${stats['volume']/1_000_000:6.1f}M"
            )
        logger.info(f"{'='*70}\n")
        
        # Збереження в кеш
        top_symbols = [s for s, _, _ in filtered[:count]]
        self.volatility_cache['symbols'] = top_symbols
        self.cache_time['symbols'] = now
        
        return top_symbols
    
    async def should_trade_symbol(self, symbol: str, min_score: float = 30.0) -> bool:
        """Перевірка чи варто торгувати символ (достатня волатильність)"""
        score = await self.calculate_volatility_score(symbol)
        return score >= min_score


# ============================================================================
# РОЗШИРЕНИЙ СПИСОК СИМВОЛІВ ДЛЯ СКАНУВАННЯ
# ============================================================================
# Ці символи автоматично завантажуються з папки models/
# Використовуємо тільки ті, для яких є натреновані моделі
try:
    from model_scanner import get_available_models
    EXTENDED_SYMBOLS = get_available_models()
except ImportError:
    # Fallback якщо model_scanner недоступний
    EXTENDED_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOTUSDT',
        'SOLUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'MATICUSDT',
        'LTCUSDT', 'ETCUSDT', 'XLMUSDT', 'ALGOUSDT', 'VETUSDT', 'FILUSDT',
        'TRXUSDT', 'DOGEUSDT'
    ]
