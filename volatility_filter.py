#!/usr/bin/env python3
"""
📊 VOLATILITY FILTER - Універсальний фільтр волатильності
Визначає, чи варто торгувати на поточних даних
Використовується в pattern_scanner та backtests
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityFilter:
    """
    Універсальний фільтр волатильності з адаптивними порогами
    
    Основні критерії:
    1. ATR (Average True Range) - мінімальна волатильність
    2. Діапазон цін - величина коливань
    3. BB Squeeze - зони стиснення перед рухом
    4. Volume trend - зростання об'єму
    5. Expansion zones - наближення великих рухів
    
    Usage:
        filter = VolatilityFilter(min_score=50.0)
        is_tradeable, score, details = filter.is_tradeable(df)
        
        if is_tradeable:
            # Відкриваємо позицію
            pass
    """
    
    def __init__(
        self,
        min_score: float = 50.0,
        min_atr_pct: float = 0.5,
        min_range_pct: float = 2.0,
        use_adaptive: bool = True,
        lookback_period: int = 100
    ):
        """
        Args:
            min_score: Мінімальний скор для торгівлі (0-100)
            min_atr_pct: Мінімальний ATR % від ціни (fixed threshold)
            min_range_pct: Мінімальний діапазон % (fixed threshold)
            use_adaptive: Використовувати адаптивні пороги
            lookback_period: Період для розрахунку адаптивних порогів
        """
        self.min_score = min_score
        self.min_atr_pct = min_atr_pct
        self.min_range_pct = min_range_pct
        self.use_adaptive = use_adaptive
        self.lookback_period = lookback_period
        
    def is_tradeable(
        self, 
        df: pd.DataFrame,
        return_details: bool = True
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Перевіряє, чи достатня волатильність для торгівлі
        
        Args:
            df: DataFrame з OHLCV та індикаторами
            return_details: Повертати деталі аналізу
        
        Returns:
            (is_tradeable, score, details)
            - is_tradeable: bool - чи можна торгувати
            - score: float (0-100) - оцінка волатильності
            - details: Dict - деталі аналізу (якщо return_details=True)
        """
        if len(df) < 50:
            return False, 0.0, {'error': 'Not enough data'} if return_details else None
        
        # Розраховуємо всі необхідні індикатори
        df = self._ensure_indicators(df)
        
        # Розраховуємо адаптивні пороги
        if self.use_adaptive:
            thresholds = self._calculate_adaptive_thresholds(df)
        else:
            thresholds = {
                'min_atr_pct': self.min_atr_pct,
                'min_range_pct': self.min_range_pct
            }
        
        # Розраховуємо скор волатильності
        score_data = self._calculate_movement_potential(df, thresholds)
        score = score_data['total_score']
        
        # Базові фільтри (швидке відсіювання)
        basic_checks = self._basic_volatility_checks(df, thresholds)
        
        # Перевірка консолідації після pump/dump
        is_consolidation = self._is_post_pump_consolidation(df)
        
        # Фінальне рішення
        is_tradeable = (
            score >= self.min_score and
            basic_checks['passed'] and
            not is_consolidation
        )
        
        details = None
        if return_details:
            details = {
                'score': score,
                'is_tradeable': is_tradeable,
                'thresholds': thresholds,
                'basic_checks': basic_checks,
                'is_consolidation': is_consolidation,
                'score_breakdown': score_data,
                'recommendation': self._get_recommendation(score, is_consolidation)
            }
        
        return is_tradeable, score, details
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Перевіряє наявність необхідних індикаторів і розраховує при потребі"""
        df = df.copy()
        
        # ATR
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Bollinger Bands
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Volume ratio
        if 'volume_ratio' not in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_adaptive_thresholds(self, df: pd.DataFrame) -> Dict:
        """
        Розраховує адаптивні пороги на основі історії активу
        """
        lookback = min(self.lookback_period, len(df))
        recent_data = df.tail(lookback)
        
        # ATR порогу - 30% від 90-перцентиля
        atr_90_percentile = recent_data['atr'].quantile(0.9)
        avg_price = recent_data['close'].mean()
        adaptive_atr_pct = (atr_90_percentile / avg_price) * 100 * 0.3
        
        # Range пороги - медіана діапазону за період
        ranges = []
        for i in range(max(50, len(df) - lookback), len(df)):
            window = df.iloc[max(0, i-50):i]
            if len(window) >= 20:
                price_range = window['high'].max() - window['low'].min()
                avg_price_window = window['close'].mean()
                range_pct = (price_range / avg_price_window) * 100
                ranges.append(range_pct)
        
        adaptive_range_pct = np.median(ranges) if ranges else self.min_range_pct
        
        # Не використовуємо занадто низькі пороги
        adaptive_atr_pct = max(adaptive_atr_pct, 0.3)
        adaptive_range_pct = max(adaptive_range_pct, 1.5)
        
        return {
            'min_atr_pct': adaptive_atr_pct,
            'min_range_pct': adaptive_range_pct,
            'historical_atr_90': atr_90_percentile,
            'is_adaptive': True
        }
    
    def _basic_volatility_checks(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """
        Базові перевірки волатильності (швидке відсіювання)
        """
        last_atr = df['atr'].iloc[-1]
        avg_price = df['close'].iloc[-50:].mean()
        atr_pct = (last_atr / avg_price) * 100
        
        # Діапазон цін за останні 50 свічок
        price_range = df['high'].iloc[-50:].max() - df['low'].iloc[-50:].min()
        price_range_pct = (price_range / avg_price) * 100
        
        # Перевірки
        atr_check = atr_pct >= thresholds['min_atr_pct']
        range_check = price_range_pct >= thresholds['min_range_pct']
        
        return {
            'passed': atr_check and range_check,
            'atr_pct': atr_pct,
            'atr_threshold': thresholds['min_atr_pct'],
            'atr_check': atr_check,
            'range_pct': price_range_pct,
            'range_threshold': thresholds['min_range_pct'],
            'range_check': range_check
        }
    
    def _is_post_pump_consolidation(self, df: pd.DataFrame) -> bool:
        """
        Детектує консолідацію після pump/dump
        Ігноруємо флет після сильного руху
        """
        if len(df) < 50:
            return False
        
        avg_price = df['close'].iloc[-50:].mean()
        
        # Діапазон останніх 15 свічок
        recent_range = df['high'].iloc[-15:].max() - df['low'].iloc[-15:].min()
        recent_range_pct = (recent_range / avg_price) * 100
        
        # Загальний діапазон за 50 свічок
        total_range = df['high'].iloc[-50:].max() - df['low'].iloc[-50:].min()
        total_range_pct = (total_range / avg_price) * 100
        
        # Якщо останні 15 свічок < 1.5% діапазон, але загальний > 5%
        if recent_range_pct < 1.5 and total_range_pct > 5:
            # Перевіряємо, чи був памп/дамп
            recent_high = df['high'].iloc[-15:].max()
            recent_low = df['low'].iloc[-15:].min()
            prev_high = df['high'].iloc[-50:-15].max()
            prev_low = df['low'].iloc[-50:-15].min()
            
            # Якщо зараз ціна далеко від попереднього діапазону
            if recent_high > prev_high * 1.05 or recent_low < prev_low * 0.95:
                return True  # Консолідація після сильного руху
        
        return False
    
    def _calculate_movement_potential(
        self, 
        df: pd.DataFrame, 
        thresholds: Dict
    ) -> Dict:
        """
        Оцінює потенціал майбутнього руху (0-100)
        
        Returns:
            Dict з розбивкою скорів по категоріях
        """
        scores = {}
        
        # 1. BB Squeeze - зони стиснення (30 балів)
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        current_width = bb_width.iloc[-1]
        avg_width = bb_width.iloc[-100:].mean()
        
        if current_width < avg_width * 0.5:
            scores['bb_squeeze'] = 30  # Сильне стиснення
        elif current_width < avg_width * 0.7:
            scores['bb_squeeze'] = 20  # Помірне стиснення
        elif current_width < avg_width * 0.9:
            scores['bb_squeeze'] = 10  # Легке стиснення
        else:
            scores['bb_squeeze'] = 0
        
        # 2. Volume Trend - зростання об'єму (25 балів)
        volume_ratio = df['volume_ratio'].iloc[-5:].mean()
        scores['volume'] = min(25, volume_ratio * 12.5)
        
        # 3. RSI в зоні розвороту (20 балів)
        rsi = df['rsi'].iloc[-1]
        if rsi < 25 or rsi > 75:
            scores['rsi'] = 20  # Сильна зона
        elif rsi < 30 or rsi > 70:
            scores['rsi'] = 15  # Помірна зона
        elif rsi < 35 or rsi > 65:
            scores['rsi'] = 10  # Легка зона
        else:
            scores['rsi'] = 5  # Нейтрально
        
        # 4. ATR Trend - зростання волатильності (15 балів)
        atr_recent = df['atr'].iloc[-5:].mean()
        atr_older = df['atr'].iloc[-20:-5].mean()
        if atr_older > 0:
            atr_trend = atr_recent / atr_older
            if atr_trend > 1.3:
                scores['atr_trend'] = 15
            elif atr_trend > 1.15:
                scores['atr_trend'] = 10
            elif atr_trend > 1.05:
                scores['atr_trend'] = 5
            else:
                scores['atr_trend'] = 0
        else:
            scores['atr_trend'] = 0
        
        # 5. Proximity to Key Levels - близькість до рівнів (10 балів)
        price = df['close'].iloc[-1]
        support = df['low'].rolling(50).min().iloc[-1]
        resistance = df['high'].rolling(50).max().iloc[-1]
        
        dist_to_support = abs(price - support) / price
        dist_to_resistance = abs(price - resistance) / price
        
        if min(dist_to_support, dist_to_resistance) < 0.01:
            scores['key_levels'] = 10  # Дуже близько
        elif min(dist_to_support, dist_to_resistance) < 0.02:
            scores['key_levels'] = 7   # Близько
        elif min(dist_to_support, dist_to_resistance) < 0.03:
            scores['key_levels'] = 4   # Помірно близько
        else:
            scores['key_levels'] = 0
        
        total_score = sum(scores.values())
        
        return {
            'total_score': total_score,
            'bb_squeeze': scores['bb_squeeze'],
            'volume': scores['volume'],
            'rsi': scores['rsi'],
            'atr_trend': scores['atr_trend'],
            'key_levels': scores['key_levels'],
            'bb_width_ratio': current_width / avg_width if avg_width > 0 else 1.0,
            'volume_ratio_avg': volume_ratio,
            'rsi_value': rsi
        }
    
    def _get_recommendation(self, score: float, is_consolidation: bool) -> str:
        """Повертає рекомендацію на основі скору"""
        if is_consolidation:
            return "🔴 SKIP - Консолідація після pump/dump"
        elif score >= 80:
            return "🟢 EXCELLENT - Дуже високий потенціал руху"
        elif score >= 65:
            return "🟢 GOOD - Хороші умови для торгівлі"
        elif score >= 50:
            return "🟡 FAIR - Помірні умови"
        elif score >= 35:
            return "🟠 POOR - Низька волатильність"
        else:
            return "🔴 SKIP - Недостатня волатильність"
    
    def get_filter_summary(self, df: pd.DataFrame) -> str:
        """
        Повертає текстовий звіт про стан волатильності
        """
        is_tradeable, score, details = self.is_tradeable(df)
        
        if not details:
            return "❌ Помилка аналізу"
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           📊 VOLATILITY FILTER ANALYSIS                      ║
╚══════════════════════════════════════════════════════════════╝

📈 Overall Score: {score:.1f}/100 {'✅' if is_tradeable else '❌'}
🎯 Recommendation: {details['recommendation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Score Breakdown:
  • BB Squeeze:    {details['score_breakdown']['bb_squeeze']:.0f}/30 pts
  • Volume:        {details['score_breakdown']['volume']:.0f}/25 pts
  • RSI Zone:      {details['score_breakdown']['rsi']:.0f}/20 pts
  • ATR Trend:     {details['score_breakdown']['atr_trend']:.0f}/15 pts
  • Key Levels:    {details['score_breakdown']['key_levels']:.0f}/10 pts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Basic Checks:
  • ATR: {details['basic_checks']['atr_pct']:.2f}% (min: {details['basic_checks']['atr_threshold']:.2f}%) {'✅' if details['basic_checks']['atr_check'] else '❌'}
  • Range: {details['basic_checks']['range_pct']:.2f}% (min: {details['basic_checks']['range_threshold']:.2f}%) {'✅' if details['basic_checks']['range_check'] else '❌'}
  • Consolidation: {'🔴 YES' if details['is_consolidation'] else '✅ NO'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Details:
  • BB Width Ratio: {details['score_breakdown']['bb_width_ratio']:.2f}
  • Volume Ratio: {details['score_breakdown']['volume_ratio_avg']:.2f}
  • RSI: {details['score_breakdown']['rsi_value']:.1f}
  • Adaptive Thresholds: {'Yes' if details['thresholds'].get('is_adaptive') else 'No'}

╚══════════════════════════════════════════════════════════════╝
"""
        return summary


# Допоміжна функція для швидкого використання
def quick_volatility_check(df: pd.DataFrame, min_score: float = 50.0) -> bool:
    """
    Швидка перевірка волатильності без детальної інформації
    
    Args:
        df: DataFrame з OHLCV даними
        min_score: Мінімальний скор
    
    Returns:
        True якщо можна торгувати
    """
    filter = VolatilityFilter(min_score=min_score)
    is_tradeable, _, _ = filter.is_tradeable(df, return_details=False)
    return is_tradeable
