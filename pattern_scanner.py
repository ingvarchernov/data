#!/usr/bin/env python3
"""
🔍 PATTERN SCANNER - Сканер технічних паттернів
Виявляє класичні паттерни Price Action та технічного аналізу
"""
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from config import FUTURES_CONFIG
from unified_binance_loader import UnifiedBinanceLoader
from volatility_filter import VolatilityFilter

# Спроба імпорту Rust pattern detector
try:
    # Спробуємо різні варіанти імпорту
    try:
        from pattern_detector import detect_patterns, detect_latest_pattern
        RUST_PATTERN_DETECTOR_AVAILABLE = True
    except ImportError:
        from pattern_detector.pattern_detector import create_detector
        RUST_PATTERN_DETECTOR_AVAILABLE = True
    
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("🦀 Rust Pattern Detector доступний - прискорення в 50x!")
except ImportError as e:
    RUST_PATTERN_DETECTOR_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"⚠️ Rust Pattern Detector недоступний, використовую Python: {e}")

logging.basicConfig(
    level=logging.INFO,  # Back to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Типи паттернів"""
    # Розворотні паттерни
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    HEAD_SHOULDERS = "Head & Shoulders"
    INVERSE_HEAD_SHOULDERS = "Inverse Head & Shoulders"
    
    # Продовжуючі паттерни
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    
    # Свічкові паттерни
    HAMMER = "Hammer"
    SHOOTING_STAR = "Shooting Star"
    ENGULFING_BULLISH = "Bullish Engulfing"
    ENGULFING_BEARISH = "Bearish Engulfing"
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    DOJI = "Doji"
    
    # Трендові паттерни
    STRONG_UPTREND = "Strong Uptrend"
    STRONG_DOWNTREND = "Strong Downtrend"
    BREAKOUT_UP = "Breakout Upward"
    BREAKOUT_DOWN = "Breakout Downward"


class SignalStrength(Enum):
    """Сила сигналу"""
    WEAK = "🟡 Слабкий"
    MEDIUM = "🟠 Середній"
    STRONG = "🔴 Сильний"
    VERY_STRONG = "🔥 Дуже сильний"


@dataclass
class Pattern:
    """Виявлений паттерн"""
    symbol: str
    pattern_type: PatternType
    strength: SignalStrength
    direction: str  # LONG/SHORT
    price: float
    timestamp: datetime
    confidence: float  # 0-100
    details: Dict
    
    def __str__(self):
        return f"{self.strength.value} {self.pattern_type.value} | {self.symbol} @ ${self.price:.2f} | Conf: {self.confidence:.0f}%"


class PatternScanner:
    """Сканер технічних паттернів з Rust прискоренням"""
    
    def __init__(self, use_rust: bool = True, min_volatility_score: float = 50.0):
        """
        Args:
            use_rust: Використовувати Rust pattern detector (якщо доступний)
            min_volatility_score: Мінімальний скор волатильності для торгівлі
        """
        self.loader = UnifiedBinanceLoader(testnet=False)
        self.symbols = FUTURES_CONFIG['symbols']
        
        # Rust pattern detector
        self.use_rust = use_rust and RUST_PATTERN_DETECTOR_AVAILABLE
        self.rust_detector = None
        if self.use_rust:
            try:
                # Імпортуємо функції напряму
                from pattern_detector import detect_patterns
                self.rust_detector = detect_patterns
                logger.info("🦀 Використовую Rust Pattern Detector (50x швидше)")
            except Exception as e:
                logger.warning(f"⚠️ Не вдалося ініціалізувати Rust detector: {e}")
                self.use_rust = False
        
        if not self.use_rust:
            logger.info("🐍 Використовую Python Pattern Detection")
        
        # Volatility filter
        self.volatility_filter = VolatilityFilter(
            min_score=min_volatility_score,
            use_adaptive=True
        )
        logger.info(f"📊 Volatility Filter активовано (min_score={min_volatility_score})")
        
    async def scan_all_symbols(self) -> List[Pattern]:
        """Сканує всі символи на паттерни"""
        all_patterns = []
        
        logger.info(f"🔍 Початок сканування {len(self.symbols)} символів...")
        
        for symbol in self.symbols:
            try:
                patterns = await self.scan_symbol(symbol)
                all_patterns.extend(patterns)
                
                if patterns:
                    logger.info(f"✅ {symbol}: знайдено {len(patterns)} паттернів")
                    
            except Exception as e:
                logger.error(f"❌ Помилка сканування {symbol}: {e}")
                
        return all_patterns
    
    async def scan_symbol(self, symbol: str, skip_volatility_check: bool = False) -> List[Pattern]:
        """Сканує один символ
        
        Args:
            symbol: Назва символу (напр. BTCUSDT)
            skip_volatility_check: Якщо True, пропустить перевірку волатильності
        """
        # Завантажуємо дані
        df = await self.loader.get_historical_data(
            symbol=symbol,
            interval='15m',
            days_back=7
        )
        
        if df is None or len(df) < 100:
            return []
        
        # Переносимо timestamp з індексу в колонку
        df = df.reset_index()
        
        # Розраховуємо індикатори
        df = self._calculate_indicators(df)
        
        # ⭐ НОВИЙ VOLATILITY FILTER - замість старого _is_tradeable
        # (пропускаємо якщо символ вже пройшов Phase 1)
        if not skip_volatility_check:
            is_tradeable, volatility_score, filter_details = self.volatility_filter.is_tradeable(df)
            
            if not is_tradeable:
                logger.debug(
                    f"⏭️ {symbol}: пропущено через низьку волатильність "
                    f"(score={volatility_score:.1f})"
                )
                return []
            
            logger.debug(f"✅ {symbol}: волатільність OK (score={volatility_score:.1f})")
        else:
            logger.debug(f"⏩ {symbol}: пропущено перевірку волатільності (skip=True)")
        
        patterns = []
        
        # ⭐ RUST PATTERN DETECTION - якщо доступний
        if self.use_rust and self.rust_detector:
            rust_patterns = self._detect_rust_patterns(df, symbol)
            logger.debug(f"  Rust: {len(rust_patterns)} patterns")
            patterns.extend(rust_patterns)
        
        # Python pattern detection (завжди як backup або доповнення)
        candlestick = self._detect_candlestick_patterns(df, symbol)
        logger.debug(f"  Candlestick: {len(candlestick)} patterns")
        patterns.extend(candlestick)
        
        trend = self._detect_trend_patterns(df, symbol)
        logger.debug(f"  Trend: {len(trend)} patterns")
        patterns.extend(trend)
        
        chart = self._detect_chart_patterns(df, symbol)
        logger.debug(f"  Chart: {len(chart)} patterns")
        patterns.extend(chart)
        
        breakout = self._detect_breakout_patterns(df, symbol)
        logger.debug(f"  Breakout: {len(breakout)} patterns")
        patterns.extend(breakout)
        
        logger.debug(f"  Total before filter: {len(patterns)}")
        
        # ФІЛЬТРУЄМО слабкі паттерни
        patterns = self._filter_quality_patterns(patterns, df)
        
        logger.debug(f"  Total after filter: {len(patterns)}")
        
        return patterns
    
    def _detect_rust_patterns(self, df: pd.DataFrame, symbol: str) -> List[Pattern]:
        """
        Виявляє паттерни через Rust (швидко!)
        """
        if not self.rust_detector:
            return []
        
        patterns = []
        
        try:
            # Підготовка даних для Rust
            opens = df['open'].values.tolist()
            highs = df['high'].values.tolist()
            lows = df['low'].values.tolist()
            closes = df['close'].values.tolist()
            
            # Виклик Rust детектора (це функція, не клас)
            window = 10
            rust_patterns = self.rust_detector(opens, highs, lows, closes, window)
            
            # Конвертація результатів у наші Pattern об'єкти
            for rp in rust_patterns:
                pattern_name = rp.get('pattern_name', 'Unknown')
                confidence = rp.get('confidence', 50.0)
                direction = rp.get('direction', 'LONG')
                index = rp.get('index', len(df) - 1)
                
                # Маппінг Rust паттернів на наші PatternType
                pattern_type = self._map_rust_pattern_type(pattern_name)
                if not pattern_type:
                    continue
                
                # Визначення сили сигналу
                if confidence >= 85:
                    strength = SignalStrength.VERY_STRONG
                elif confidence >= 75:
                    strength = SignalStrength.STRONG
                elif confidence >= 65:
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK
                
                # Отримуємо дані свічки
                candle = df.iloc[index]
                
                patterns.append(Pattern(
                    symbol=symbol,
                    pattern_type=pattern_type,
                    strength=strength,
                    direction=direction,
                    price=candle['close'],
                    timestamp=candle['timestamp'],
                    confidence=confidence,
                    details={
                        'source': 'Rust',
                        'pattern_name': pattern_name,
                        'index': index,
                        'rsi': candle.get('rsi', 50)
                    }
                ))
                
            if patterns:
                logger.debug(f"🦀 Rust detector: знайдено {len(patterns)} паттернів для {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Помилка Rust pattern detection для {symbol}: {e}")
        
        return patterns
    
    def _map_rust_pattern_type(self, rust_pattern_name: str) -> Optional[PatternType]:
        """Маппінг назв паттернів з Rust на наші PatternType"""
        mapping = {
            'Double Top': PatternType.DOUBLE_TOP,
            'Double Bottom': PatternType.DOUBLE_BOTTOM,
            'Head and Shoulders': PatternType.HEAD_SHOULDERS,
            'Inverse Head and Shoulders': PatternType.INVERSE_HEAD_SHOULDERS,
            'Symmetrical Triangle': PatternType.SYMMETRICAL_TRIANGLE,
            'Flag': PatternType.BULL_FLAG,  # Rust може не розрізняти bull/bear
            'Pennant': PatternType.BULL_FLAG,
            'Rising Wedge': PatternType.ASCENDING_TRIANGLE,
            'Falling Wedge': PatternType.DESCENDING_TRIANGLE,
            'Three White Soldiers': PatternType.STRONG_UPTREND,
            'Three Black Crows': PatternType.STRONG_DOWNTREND,
        }
        return mapping.get(rust_pattern_name)
        df = self._calculate_indicators(df)
        
        # Перевірка якості даних та волатильності
        if not self._is_tradeable(df):
            return []
        
        patterns = []
        
        # Виявляємо різні типи паттернів
        patterns.extend(self._detect_candlestick_patterns(df, symbol))
        patterns.extend(self._detect_trend_patterns(df, symbol))
        patterns.extend(self._detect_chart_patterns(df, symbol))
        patterns.extend(self._detect_breakout_patterns(df, symbol))
        
        # ФІЛЬТРУЄМО слабкі паттерни
        patterns = self._filter_quality_patterns(patterns, df)
        
        return patterns
    
    def _filter_quality_patterns(self, patterns: List[Pattern], df: pd.DataFrame) -> List[Pattern]:
        """
        Фільтрує паттерни за якістю
        Залишає тільки найсильніші та найбільш значущі
        """
        if not patterns:
            return []
        
        filtered = []
        rejected_low_conf = 0
        rejected_final = 0
        
        for pattern in patterns:
            # 1. Мінімальна впевненість 60%
            if pattern.confidence < 60:
                rejected_low_conf += 1
                continue
            
            # 2. Перевірка об'єму для сильних паттернів
            last_volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Сильні паттерни мають бути підтверджені об'ємом
            if pattern.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                if last_volume_ratio < 1.0:  # Об'єм нижче середнього
                    # Знижуємо силу
                    pattern.strength = SignalStrength.MEDIUM
                    pattern.confidence *= 0.8
            
            # 3. Перевірка RSI для підтвердження
            last_rsi = df['rsi'].iloc[-1]
            
            if pattern.direction == "LONG":
                # LONG паттерн на перекупленості (RSI > 70) - підозріло
                if last_rsi > 70:
                    pattern.confidence *= 0.7
                # LONG на перепроданості (RSI < 30) - дуже добре
                elif last_rsi < 30:
                    pattern.confidence = min(95, pattern.confidence * 1.2)
                    
            elif pattern.direction == "SHORT":
                # SHORT на перепроданості (RSI < 30) - підозріло
                if last_rsi < 30:
                    pattern.confidence *= 0.7
                # SHORT на перекупленості (RSI > 70) - дуже добре
                elif last_rsi > 70:
                    pattern.confidence = min(95, pattern.confidence * 1.2)
            
            # 4. Перевірка тренду
            ema9 = df['ema9'].iloc[-1]
            ema21 = df['ema21'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            price = df['close'].iloc[-1]
            
            # LONG проти тренду (ціна нижче всіх EMA) - слабкіше
            if pattern.direction == "LONG":
                if price < ema9 < ema21 < ema50:
                    pattern.confidence *= 0.8
                # LONG за трендом - сильніше
                elif price > ema9 > ema21:
                    pattern.confidence = min(95, pattern.confidence * 1.1)
            
            # SHORT проти тренду (ціна вище всіх EMA) - слабкіше
            elif pattern.direction == "SHORT":
                if price > ema9 > ema21 > ema50:
                    pattern.confidence *= 0.8
                # SHORT за трендом - сильніше
                elif price < ema9 < ema21:
                    pattern.confidence = min(95, pattern.confidence * 1.1)
            
            # 5. Фінальний фільтр: confidence >= 50%
            if pattern.confidence >= 50:
                filtered.append(pattern)
            else:
                rejected_final += 1
        
        logger.debug(f"  Filter stats: {rejected_low_conf} rejected (<60%), {rejected_final} rejected final (<50%)")
        
        # Сортуємо за впевненістю
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        
        # Залишаємо максимум 3 найкращих паттерни на символ
        return filtered[:3]
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розраховує технічні індикатори"""
        # EMA
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR для волатильності
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame, symbol: str) -> List[Pattern]:
        """Виявляє свічкові паттерни"""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        body = abs(last['close'] - last['open'])
        body_pct = body / last['open']
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        
        # HAMMER (бичачий розворот)
        if (lower_shadow > body * 2 and 
            upper_shadow < body * 0.3 and
            last['close'] > last['open']):
            
            confidence = min(95, 60 + (lower_shadow / body) * 10)
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.HAMMER,
                strength=SignalStrength.STRONG if confidence > 75 else SignalStrength.MEDIUM,
                direction="LONG",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'body_pct': body_pct * 100,
                    'lower_shadow': lower_shadow,
                    'upper_shadow': upper_shadow,
                    'rsi': last['rsi']
                }
            ))
        
        # SHOOTING STAR (ведмежий розворот)
        if (upper_shadow > body * 2 and 
            lower_shadow < body * 0.3 and
            last['close'] < last['open']):
            
            confidence = min(95, 60 + (upper_shadow / body) * 10)
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.SHOOTING_STAR,
                strength=SignalStrength.STRONG if confidence > 75 else SignalStrength.MEDIUM,
                direction="SHORT",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'body_pct': body_pct * 100,
                    'upper_shadow': upper_shadow,
                    'lower_shadow': lower_shadow,
                    'rsi': last['rsi']
                }
            ))
        
        # BULLISH ENGULFING
        if (prev['close'] < prev['open'] and  # Попередня свічка червона
            last['close'] > last['open'] and  # Поточна зелена
            last['open'] < prev['close'] and  # Відкриття нижче закриття попередньої
            last['close'] > prev['open']):    # Закриття вище відкриття попередньої
            
            engulf_ratio = body / abs(prev['close'] - prev['open'])
            confidence = min(95, 65 + engulf_ratio * 15)
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.ENGULFING_BULLISH,
                strength=SignalStrength.STRONG if confidence > 80 else SignalStrength.MEDIUM,
                direction="LONG",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'engulf_ratio': engulf_ratio,
                    'volume_ratio': last['volume_ratio'],
                    'rsi': last['rsi']
                }
            ))
        
        # BEARISH ENGULFING
        if (prev['close'] > prev['open'] and  # Попередня свічка зелена
            last['close'] < last['open'] and  # Поточна червона
            last['open'] > prev['close'] and  # Відкриття вище закриття попередньої
            last['close'] < prev['open']):    # Закриття нижче відкриття попередньої
            
            engulf_ratio = body / abs(prev['close'] - prev['open'])
            confidence = min(95, 65 + engulf_ratio * 15)
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.ENGULFING_BEARISH,
                strength=SignalStrength.STRONG if confidence > 80 else SignalStrength.MEDIUM,
                direction="SHORT",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'engulf_ratio': engulf_ratio,
                    'volume_ratio': last['volume_ratio'],
                    'rsi': last['rsi']
                }
            ))
        
        # DOJI (невизначеність)
        if body_pct < 0.001:  # Тіло менше 0.1%
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.DOJI,
                strength=SignalStrength.WEAK,
                direction="NEUTRAL",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=70,
                details={
                    'body_pct': body_pct * 100,
                    'upper_shadow': upper_shadow,
                    'lower_shadow': lower_shadow,
                    'meaning': 'Невизначеність, очікуйте розвороту'
                }
            ))
        
        return patterns
    
    def _detect_trend_patterns(self, df: pd.DataFrame, symbol: str) -> List[Pattern]:
        """Виявляє трендові паттерни"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        last = df.iloc[-1]
        
        # Сильний бичачий тренд
        if (last['ema9'] > last['ema21'] > last['ema50'] and
            last['close'] > last['ema9'] and
            last['rsi'] > 50 and last['rsi'] < 80):
            
            # Відстань між EMA як міра сили тренду
            ema_spread = ((last['ema9'] - last['ema50']) / last['ema50']) * 100
            confidence = min(95, 70 + ema_spread * 2)
            
            strength = SignalStrength.VERY_STRONG if ema_spread > 3 else SignalStrength.STRONG
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.STRONG_UPTREND,
                strength=strength,
                direction="LONG",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'ema_alignment': 'Бичача',
                    'ema_spread_pct': ema_spread,
                    'rsi': last['rsi'],
                    'volume_ratio': last['volume_ratio']
                }
            ))
        
        # Сильний ведмежий тренд
        if (last['ema9'] < last['ema21'] < last['ema50'] and
            last['close'] < last['ema9'] and
            last['rsi'] < 50 and last['rsi'] > 20):
            
            ema_spread = ((last['ema50'] - last['ema9']) / last['ema50']) * 100
            confidence = min(95, 70 + ema_spread * 2)
            
            strength = SignalStrength.VERY_STRONG if ema_spread > 3 else SignalStrength.STRONG
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.STRONG_DOWNTREND,
                strength=strength,
                direction="SHORT",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'ema_alignment': 'Ведмежа',
                    'ema_spread_pct': ema_spread,
                    'rsi': last['rsi'],
                    'volume_ratio': last['volume_ratio']
                }
            ))
        
        return patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame, symbol: str) -> List[Pattern]:
        """Виявляє графічні паттерни (Double Top/Bottom, H&S)"""
        patterns = []
        
        if len(df) < 100:
            return patterns
        
        # Використовуємо останні 100 свічок
        recent = df.tail(100)
        last = df.iloc[-1]
        
        # Знаходимо локальні максимуми та мінімуми
        highs = recent['high'].rolling(10, center=True).apply(
            lambda x: x[len(x)//2] == x.max(), raw=True
        )
        lows = recent['low'].rolling(10, center=True).apply(
            lambda x: x[len(x)//2] == x.min(), raw=True
        )
        
        peaks = recent[highs == 1]['high'].tolist()
        troughs = recent[lows == 1]['low'].tolist()
        
        # DOUBLE TOP
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if len(last_two_peaks) == 2:
                diff = abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0]
                
                if diff < 0.02:  # Піки в межах 2%
                    confidence = min(90, 75 - diff * 1000)
                    
                    patterns.append(Pattern(
                        symbol=symbol,
                        pattern_type=PatternType.DOUBLE_TOP,
                        strength=SignalStrength.STRONG,
                        direction="SHORT",
                        price=last['close'],
                        timestamp=last['timestamp'],
                        confidence=confidence,
                        details={
                            'peak1': last_two_peaks[0],
                            'peak2': last_two_peaks[1],
                            'diff_pct': diff * 100,
                            'warning': 'Чекайте підтвердження пробою вниз'
                        }
                    ))
        
        # DOUBLE BOTTOM
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            if len(last_two_troughs) == 2:
                diff = abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0]
                
                if diff < 0.02:  # Дно в межах 2%
                    confidence = min(90, 75 - diff * 1000)
                    
                    patterns.append(Pattern(
                        symbol=symbol,
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        strength=SignalStrength.STRONG,
                        direction="LONG",
                        price=last['close'],
                        timestamp=last['timestamp'],
                        confidence=confidence,
                        details={
                            'trough1': last_two_troughs[0],
                            'trough2': last_two_troughs[1],
                            'diff_pct': diff * 100,
                            'warning': 'Чекайте підтвердження пробою вгору'
                        }
                    ))
        
        return patterns
    
    def _detect_breakout_patterns(self, df: pd.DataFrame, symbol: str) -> List[Pattern]:
        """Виявляє пробої рівнів"""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Пробій Bollinger Bands вгору
        if (prev['close'] <= prev['bb_upper'] and 
            last['close'] > last['bb_upper'] and
            last['volume_ratio'] > 1.5):
            
            confidence = min(95, 70 + (last['volume_ratio'] - 1) * 20)
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.BREAKOUT_UP,
                strength=SignalStrength.STRONG if last['volume_ratio'] > 2 else SignalStrength.MEDIUM,
                direction="LONG",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'bb_upper': last['bb_upper'],
                    'volume_ratio': last['volume_ratio'],
                    'rsi': last['rsi'],
                    'type': 'Bollinger Band Breakout'
                }
            ))
        
        # Пробій Bollinger Bands вниз
        if (prev['close'] >= prev['bb_lower'] and 
            last['close'] < last['bb_lower'] and
            last['volume_ratio'] > 1.5):
            
            confidence = min(95, 70 + (last['volume_ratio'] - 1) * 20)
            
            patterns.append(Pattern(
                symbol=symbol,
                pattern_type=PatternType.BREAKOUT_DOWN,
                strength=SignalStrength.STRONG if last['volume_ratio'] > 2 else SignalStrength.MEDIUM,
                direction="SHORT",
                price=last['close'],
                timestamp=last['timestamp'],
                confidence=confidence,
                details={
                    'bb_lower': last['bb_lower'],
                    'volume_ratio': last['volume_ratio'],
                    'rsi': last['rsi'],
                    'type': 'Bollinger Band Breakdown'
                }
            ))
        
        return patterns
    
    def print_patterns(self, patterns: List[Pattern]):
        """Виводить знайдені паттерни"""
        if not patterns:
            logger.info("❌ Паттернів не знайдено")
            return
        
        # Сортуємо за силою та впевненістю
        sorted_patterns = sorted(
            patterns, 
            key=lambda x: (x.strength.value, x.confidence),
            reverse=True
        )
        
        print("\n" + "="*80)
        print("🔍 ЗНАЙДЕНІ ПАТТЕРНИ")
        print("="*80)
        
        for i, pattern in enumerate(sorted_patterns, 1):
            print(f"\n{i}. {pattern}")
            print(f"   Напрямок: {pattern.direction}")
            print(f"   Час: {pattern.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Деталі:")
            for key, value in pattern.details.items():
                if isinstance(value, float):
                    print(f"     • {key}: {value:.2f}")
                else:
                    print(f"     • {key}: {value}")
        
        print("\n" + "="*80)
        
        # Статистика
        print("\n📊 СТАТИСТИКА:")
        total = len(sorted_patterns)
        long_count = sum(1 for p in sorted_patterns if p.direction == "LONG")
        short_count = sum(1 for p in sorted_patterns if p.direction == "SHORT")
        
        print(f"Всього паттернів: {total}")
        print(f"LONG сигналів: {long_count} ({long_count/total*100:.1f}%)")
        print(f"SHORT сигналів: {short_count} ({short_count/total*100:.1f}%)")
        
        # По типах
        pattern_types = {}
        for p in sorted_patterns:
            pattern_types[p.pattern_type.value] = pattern_types.get(p.pattern_type.value, 0) + 1
        
        print("\nПо типах паттернів:")
        for ptype, count in sorted(pattern_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {ptype}: {count}")


async def main():
    """Головна функція"""
    scanner = PatternScanner()
    
    print("\n🔍 PATTERN SCANNER - Сканер технічних паттернів")
    print("="*80)
    print("Аналізуємо ринок на наявність класичних паттернів Price Action...")
    print("="*80 + "\n")
    
    # Сканування
    patterns = await scanner.scan_all_symbols()
    
    # Виведення результатів
    scanner.print_patterns(patterns)
    
    print("\n💡 Порада: Використовуйте ці паттерни як підказки,")
    print("   але завжди підтверджуйте власним аналізом перед торгівлею!")


if __name__ == "__main__":
    asyncio.run(main())
