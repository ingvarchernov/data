"""
Pattern Analytics Module
Розрахунок точок входу, Stop Loss та Take Profit для торгових паттернів
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Import enhanced analytics
try:
    from enhanced_analytics import enhanced_analytics, MarketRegime
    ENHANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYTICS_AVAILABLE = False
    logger.warning("Enhanced analytics not available, using basic methods")
    
    # Fallback enum definition
    from enum import Enum
    class MarketRegime(Enum):
        """Ринки режими"""
        TRENDING_UP = "trending_up"
        TRENDING_DOWN = "trending_down"
        RANGING = "ranging"
        VOLATILE = "volatile"


class PatternAnalytics:
    """Аналіз паттернів з визначенням Entry, SL, TP"""
    
    # Коефіцієнти Risk/Reward для різних паттернів
    PATTERN_RR_RATIOS = {
        'Head and Shoulders': {'sl_multiplier': 1.0, 'tp_multiplier': 2.0},
        'Double Top': {'sl_multiplier': 1.0, 'tp_multiplier': 1.8},
        'Double Bottom': {'sl_multiplier': 1.0, 'tp_multiplier': 1.8},
        'Triangle': {'sl_multiplier': 0.8, 'tp_multiplier': 2.5},
        'Wedge': {'sl_multiplier': 0.9, 'tp_multiplier': 2.2},
        'Flag': {'sl_multiplier': 0.7, 'tp_multiplier': 2.0},
        'Channel': {'sl_multiplier': 1.0, 'tp_multiplier': 2.0},
        'Compression Breakout': {'sl_multiplier': 1.0, 'tp_multiplier': 3.0},  # R:R 1:3
    }
    
    def __init__(self, use_enhanced_analytics: bool = True):
        """Ініціалізація аналітики"""
        self.use_enhanced_analytics = use_enhanced_analytics and ENHANCED_ANALYTICS_AVAILABLE
    
    @staticmethod
    def smart_round(price: float, symbol: str = '') -> float:
        """
        Розумне округлення цін залежно від вартості
        
        Args:
            price: Ціна для округлення
            symbol: Символ (опціонально для додаткової логіки)
        
        Returns:
            Округлена ціна
        """
        if price >= 10000:  # BTC, ETH дорогі монети
            return round(price, 2)
        elif price >= 1000:  # Середні монети
            return round(price, 2)
        elif price >= 100:
            return round(price, 2)
        elif price >= 10:
            return round(price, 3)
        elif price >= 1:
            return round(price, 4)
        elif price >= 0.1:
            return round(price, 5)
        elif price >= 0.01:
            return round(price, 6)
        elif price >= 0.001:
            return round(price, 7)
        else:  # Дуже дешеві монети
            return round(price, 8)
    
    @staticmethod
    def calculate_price_change_pct(entry: float, target: float, direction: str = 'LONG') -> float:
        """
        Розрахунок % зміни ціни від entry до target
        
        Args:
            entry: Ціна входу
            target: Цільова ціна (TP або SL)
            direction: Напрямок позиції
        
        Returns:
            Відсоток зміни (позитивний = профіт, негативний = збиток)
        """
        if entry == 0:
            return 0.0
        
        change_pct = ((target - entry) / entry) * 100
        
        # Для SHORT позиції інвертуємо
        if direction == 'SHORT':
            change_pct = -change_pct
        
        return round(change_pct, 2)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Розрахунок Average True Range (ATR)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return 0.0
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Знаходження рівнів підтримки та опору"""
        try:
            # Використовуємо МЕНШИЙ період для більш актуальних рівнів
            recent_data = df.tail(window)  # Було window*2
            
            # Support - мінімальна ціна з недавнього періоду
            support = recent_data['low'].min()
            
            # Resistance - максимальна ціна з недавнього періоду
            resistance = recent_data['high'].max()
            
            # Обмежуємо максимальний спред (не більше 30% від поточної ціни)
            current_price = float(df['close'].iloc[-1])
            max_spread = current_price * 0.3
            
            if (resistance - current_price) > max_spread:
                resistance = current_price + max_spread
            if (current_price - support) > max_spread:
                support = current_price - max_spread
            
            return float(support), float(resistance)
        except Exception as e:
            logger.warning(f"Support/Resistance calculation error: {e}")
            current_price = float(df['close'].iloc[-1])
            return current_price * 0.98, current_price * 1.02
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Визначення режиму ринку (тренд/флет/волатильність)"""
        try:
            if len(df) < 50:
                return MarketRegime.RANGING
            
            # Останні 50 свічок
            recent = df.tail(50)
            
            # EMA alignment (тренд)
            ema9 = recent['close'].ewm(span=9).mean()
            ema21 = recent['close'].ewm(span=21).mean()
            ema50 = recent['close'].ewm(span=50).mean()
            
            # Directional movement
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            
            # Volatility (ATR)
            atr = self.calculate_atr(recent, 14)
            avg_price = recent['close'].mean()
            volatility_pct = (atr / avg_price) * 100
            
            # Trend strength
            ema_alignment_up = (ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1])
            ema_alignment_down = (ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1])
            
            # High volatility
            if volatility_pct > 3.0:
                return MarketRegime.VOLATILE
            
            # Strong uptrend
            if ema_alignment_up and price_change > 0.05:  # +5%
                return MarketRegime.TRENDING_UP
            
            # Strong downtrend
            if ema_alignment_down and price_change < -0.05:  # -5%
                return MarketRegime.TRENDING_DOWN
            
            # Ranging market
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.warning(f"Market regime detection error: {e}")
            return MarketRegime.RANGING
    
    def calculate_entry_sl_tp(
        self,
        df: pd.DataFrame,
        pattern_type: str,
        pattern_direction: str,
        pattern_data: Dict
    ) -> Dict[str, float]:
        """
        Розрахунок точок входу, SL та TP
        
        Args:
            df: DataFrame з OHLCV даними
            pattern_type: Тип паттерну (Head and Shoulders, Double Top, тощо)
            pattern_direction: Напрямок (LONG/SHORT)
            pattern_data: Додаткові дані паттерну (peak_price, valley_price, тощо)
        
        Returns:
            Dict з entry_price, stop_loss, take_profit, risk_reward_ratio
        """
        try:
            # Detect market regime for better risk management
            market_regime = self.detect_market_regime(df)
            
            # Для compression breakout використовуємо ціну з pattern_data (breakout candle)
            # Інакше беремо останню ціну в DataFrame
            if 'Compression Breakout' in pattern_type and 'breakout_price' in pattern_data:
                current_price = float(pattern_data['breakout_price'])
            else:
                current_price = float(df['close'].iloc[-1])
            
            atr = self.calculate_atr(df)
            support, resistance = self.find_support_resistance(df)
            
            # СПЕЦІАЛЬНА ЛОГІКА ДЛЯ COMPRESSION BREAKOUT
            if 'Compression Breakout' in pattern_type:
                # Використовуємо compression zone для SL
                compression_high = pattern_data.get('compression_high', current_price * 1.01)
                compression_low = pattern_data.get('compression_low', current_price * 0.99)
                
                # current_price - це breakout candle close
                breakout_price = current_price
                
                if pattern_direction == 'LONG':
                    entry_price = breakout_price * 1.002  # Entry next candle
                    stop_loss = compression_low * 0.999  # SL just below compression zone
                    
                    # Перевірка логіки
                    if stop_loss >= entry_price:
                        # Якщо compression_low вище entry (помилка), використовуємо фіксований відсоток
                        stop_loss = entry_price * 0.97
                    
                    sl_distance = entry_price - stop_loss
                    take_profit = entry_price + (sl_distance * 3.0)  # R:R 1:3
                    
                else:  # SHORT
                    entry_price = breakout_price * 0.998
                    stop_loss = compression_high * 1.001  # SL just above compression zone
                    
                    # Перевірка логіки
                    if stop_loss <= entry_price:
                        # Якщо compression_high нижче entry (помилка), використовуємо фіксований відсоток
                        stop_loss = entry_price * 1.03
                    
                    sl_distance = stop_loss - entry_price
                    take_profit = entry_price - (sl_distance * 3.0)  # R:R 1:3
                
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward_ratio = reward / risk if risk > 0 else 3.0
                
                symbol_name = pattern_data.get('symbol', '')
                entry_price = self.smart_round(entry_price, symbol_name)
                stop_loss = self.smart_round(stop_loss, symbol_name)
                take_profit = self.smart_round(take_profit, symbol_name)
                current_price = self.smart_round(current_price, symbol_name)
                
                tp_profit_pct = self.calculate_price_change_pct(entry_price, take_profit, pattern_direction)
                sl_loss_pct = self.calculate_price_change_pct(entry_price, stop_loss, pattern_direction)
                
                return {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'current_price': current_price,
                    'direction': pattern_direction,
                    'risk_reward_ratio': round(risk_reward_ratio, 2),
                    'tp_profit_pct': tp_profit_pct,
                    'sl_loss_pct': sl_loss_pct,
                    'atr': self.smart_round(atr, symbol_name),
                    'support': self.smart_round(support, symbol_name),
                    'resistance': self.smart_round(resistance, symbol_name),
                }
            
            # Отримуємо коефіцієнти для паттерну
            pattern_key = next((k for k in self.PATTERN_RR_RATIOS.keys() if k in pattern_type), None)
            if pattern_key:
                rr_config = self.PATTERN_RR_RATIOS[pattern_key].copy()
            else:
                # Дефолтні значення
                rr_config = {'sl_multiplier': 1.0, 'tp_multiplier': 2.0}
            
            # Adjust multipliers based on market regime
            if market_regime == MarketRegime.TRENDING_UP:
                rr_config['tp_multiplier'] *= 1.2  # More aggressive TP in uptrend
                rr_config['sl_multiplier'] *= 0.8  # Tighter SL
            elif market_regime == MarketRegime.TRENDING_DOWN:
                rr_config['tp_multiplier'] *= 1.2  # More aggressive TP in downtrend
                rr_config['sl_multiplier'] *= 0.8  # Tighter SL
            elif market_regime == MarketRegime.VOLATILE:
                rr_config['sl_multiplier'] *= 1.3  # Wider SL in volatile markets
                rr_config['tp_multiplier'] *= 0.8  # More conservative TP
            # Ranging markets use default values
            
            sl_multiplier = rr_config['sl_multiplier']
            tp_multiplier = rr_config['tp_multiplier']
            
            # Розрахунок для LONG позицій
            if pattern_direction == 'LONG':
                # Entry трохи вище поточної ціни (підтвердження)
                entry_price = current_price * 1.002  # +0.2%
                
                # SL нижче support або ATR
                if support > 0 and support < current_price:
                    # Використовуємо support як базу для SL
                    sl_distance = (current_price - support) * sl_multiplier
                    stop_loss = current_price - sl_distance
                else:
                    # Використовуємо ATR
                    stop_loss = current_price - (atr * sl_multiplier * 1.5)
                
                # TP вище resistance або на основі R/R
                sl_risk = entry_price - stop_loss
                take_profit = entry_price + (sl_risk * tp_multiplier)
                
                # Обмеження: TP не повинен бути дуже далеко (макс 2x від entry)
                max_tp = entry_price * 2.0
                if take_profit > max_tp:
                    take_profit = max_tp
                
                # Коригуємо TP якщо є resistance
                if resistance > entry_price:
                    potential_tp = resistance * 0.99  # трохи нижче resistance
                    if potential_tp > entry_price:
                        # Використовуємо найближчий варіант
                        take_profit = min(take_profit, potential_tp)
            
            # Розрахунок для SHORT позицій
            else:  # SHORT
                # Entry трохи нижче поточної ціни
                entry_price = current_price * 0.998  # -0.2%
                
                # SL вище resistance або ATR
                if resistance > current_price:
                    # Використовуємо resistance як базу для SL
                    sl_distance = (resistance - current_price) * sl_multiplier
                    stop_loss = current_price + sl_distance
                else:
                    # Використовуємо ATR
                    stop_loss = current_price + (atr * sl_multiplier * 1.5)
                
                # TP нижче support або на основі R/R
                sl_risk = stop_loss - entry_price
                take_profit = entry_price - (sl_risk * tp_multiplier)
                
                # Обмеження: TP не повинен бути дуже далеко (макс 10% профіту)
                min_tp = entry_price * 0.90
                if take_profit < min_tp:
                    take_profit = min_tp
                
                # Коригуємо TP якщо є support
                if support < entry_price:
                    potential_tp = support * 1.01  # трохи вище support
                    if potential_tp < entry_price:
                        # Використовуємо найближчий варіант
                        take_profit = max(take_profit, potential_tp)
            
            # Розрахунок R/R співвідношення
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Додаткові перевірки
            if pattern_direction == 'LONG':
                # SL має бути нижче entry
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.97
                # TP має бути вище entry
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.04
            else:  # SHORT
                # SL має бути вище entry
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.03
                # TP має бути нижче entry
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.96
            
            # Розумне округлення
            symbol_name = pattern_data.get('symbol', '')
            entry_price = self.smart_round(entry_price, symbol_name)
            stop_loss = self.smart_round(stop_loss, symbol_name)
            take_profit = self.smart_round(take_profit, symbol_name)
            current_price = self.smart_round(current_price, symbol_name)
            
            # Розрахунок відсотків
            tp_profit_pct = self.calculate_price_change_pct(entry_price, take_profit, pattern_direction)
            sl_loss_pct = self.calculate_price_change_pct(entry_price, stop_loss, pattern_direction)
            
            # Enhanced Analytics Integration
            enhanced_confidence = pattern_data.get('confidence', 50)
            market_regime = None
            ensemble_signal = None
            
            if self.use_enhanced_analytics:
                try:
                    # Prepare pattern data for enhanced analysis
                    enhanced_pattern_data = {
                        'pattern_type': pattern_type,
                        'direction': pattern_direction,
                        'confidence': pattern_data.get('confidence', 50),
                        **pattern_data
                    }
                    
                    # Run enhanced analysis
                    analysis_result = enhanced_analytics.analyze_pattern(df, enhanced_pattern_data)
                    
                    # Update confidence with enhanced scoring
                    enhanced_confidence = analysis_result.get('enhanced_confidence', enhanced_confidence)
                    market_regime = analysis_result.get('regime')
                    ensemble_signal = analysis_result.get('ensemble_signal')
                    
                    # Adjust SL/TP based on market regime and ML predictions
                    if ensemble_signal and ensemble_signal.final_score > 60:
                        # In trending markets, allow wider stops for better R:R
                        if market_regime in ['trending_up', 'trending_down']:
                            if pattern_direction == 'LONG':
                                stop_loss *= 0.98  # Slightly wider stop
                                take_profit *= 1.05  # Higher target
                            else:
                                stop_loss *= 1.02  # Slightly wider stop
                                take_profit *= 0.95  # Lower target
                        
                        # In volatile markets, use tighter stops
                        elif market_regime == 'volatile':
                            if pattern_direction == 'LONG':
                                stop_loss = entry_price - (entry_price - stop_loss) * 0.8
                            else:
                                stop_loss = entry_price + (stop_loss - entry_price) * 0.8
                    
                except Exception as e:
                    logger.warning(f"Enhanced analytics integration error: {e}")
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'current_price': current_price,
                'direction': pattern_direction,
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'tp_profit_pct': tp_profit_pct,
                'sl_loss_pct': sl_loss_pct,
                'atr': self.smart_round(atr, symbol_name),
                'support': self.smart_round(support, symbol_name),
                'resistance': self.smart_round(resistance, symbol_name),
                'enhanced_confidence': enhanced_confidence,
                'market_regime': market_regime,
                'ensemble_signal': ensemble_signal,
            }
            
        except Exception as e:
            logger.error(f"Error calculating entry/SL/TP: {e}")
            # Повертаємо консервативні значення
            current_price = float(df['close'].iloc[-1])
            symbol_name = pattern_data.get('symbol', '')
            
            if pattern_direction == 'LONG':
                entry = self.smart_round(current_price * 1.002, symbol_name)
                sl = self.smart_round(current_price * 0.97, symbol_name)
                tp = self.smart_round(current_price * 1.06, symbol_name)
                tp_pct = self.calculate_price_change_pct(entry, tp, 'LONG')
                sl_pct = self.calculate_price_change_pct(entry, sl, 'LONG')
                
                return {
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'current_price': self.smart_round(current_price, symbol_name),
                    'direction': 'LONG',
                    'risk_reward_ratio': 2.0,
                    'tp_profit_pct': tp_pct,
                    'sl_loss_pct': sl_pct,
                    'atr': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                }
            else:
                entry = self.smart_round(current_price * 0.998, symbol_name)
                sl = self.smart_round(current_price * 1.03, symbol_name)
                tp = self.smart_round(current_price * 0.94, symbol_name)
                tp_pct = self.calculate_price_change_pct(entry, tp, 'SHORT')
                sl_pct = self.calculate_price_change_pct(entry, sl, 'SHORT')
                
                return {
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'current_price': self.smart_round(current_price, symbol_name),
                    'direction': 'SHORT',
                    'risk_reward_ratio': 2.0,
                    'tp_profit_pct': tp_pct,
                    'sl_loss_pct': sl_pct,
                    'atr': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                }
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 1.0,
        max_position_percent: float = 10.0
    ) -> Dict[str, float]:
        """
        Розрахунок розміру позиції на основі ризик-менеджменту
        
        Args:
            balance: Баланс акаунта (в USDT)
            entry_price: Ціна входу
            stop_loss: Ціна стоп-лосу
            risk_percent: Макс. % від балансу на ризик (default 1%)
            max_position_percent: Макс. % від балансу на позицію (default 10%)
        
        Returns:
            Dict з quantity, position_value, risk_amount
        """
        try:
            # Максимальна сума ризику
            risk_amount = balance * (risk_percent / 100)
            
            # Відстань до SL в %
            sl_distance = abs(entry_price - stop_loss) / entry_price
            
            # Розмір позиції базовий на ризику
            position_value = risk_amount / sl_distance if sl_distance > 0 else 0
            
            # Обмежуємо позицію max_position_percent від балансу
            max_position_value = balance * (max_position_percent / 100)
            position_value = min(position_value, max_position_value)
            
            # Кількість токенів
            quantity = position_value / entry_price if entry_price > 0 else 0
            
            # Фактичний ризик
            actual_risk = position_value * sl_distance
            
            return {
                'quantity': round(quantity, 8),
                'position_value': round(position_value, 2),
                'risk_amount': round(actual_risk, 2),
                'risk_percent': round((actual_risk / balance) * 100, 2) if balance > 0 else 0,
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'quantity': 0.0,
                'position_value': 0.0,
                'risk_amount': 0.0,
                'risk_percent': 0.0,
            }


# Singleton instance
_analytics = None


def get_analytics() -> PatternAnalytics:
    """Отримати singleton instance аналітики"""
    global _analytics
    if _analytics is None:
        _analytics = PatternAnalytics()
    return _analytics


if __name__ == "__main__":
    # Тестування
    logging.basicConfig(level=logging.INFO)
    
    # Створюємо тестові дані
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
    })
    
    analytics = get_analytics()
    
    # Тест для LONG
    result_long = analytics.calculate_entry_sl_tp(
        test_data,
        'Double Bottom',
        'LONG',
        {}
    )
    print("\n📈 LONG Position Analysis:")
    for key, value in result_long.items():
        print(f"  {key}: {value}")
    
    # Тест для SHORT
    result_short = analytics.calculate_entry_sl_tp(
        test_data,
        'Double Top',
        'SHORT',
        {}
    )
    print("\n📉 SHORT Position Analysis:")
    for key, value in result_short.items():
        print(f"  {key}: {value}")
    
    # Тест position sizing
    balance = 1000.0  # $1000
    position = analytics.calculate_position_size(
        balance,
        result_long['entry_price'],
        result_long['stop_loss'],
        risk_percent=1.0
    )
    print("\n💰 Position Sizing (1% risk):")
    for key, value in position.items():
        print(f"  {key}: {value}")
