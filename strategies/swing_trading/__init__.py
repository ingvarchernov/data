"""
Стратегія свінг-трейдингу - позиції на 1-5 днів
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from strategies.base import TradingStrategy, TradeSignal, TradeAction, Position


class SwingTradingStrategy(TradingStrategy):
    """
    Стратегія свінг-трейдингу для середньострокових позицій

    Особливості:
    - Тривалість позицій: 1-5 днів
    - Цілі: 5-15% прибуток на угоду
    - Фокус на трендових рухах
    - Менша частота угод, вища точність
    """

    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        super().__init__("swing_trading", symbols, config)

        # Параметри свінг-трейдингу
        self.min_confidence = 0.15  # Підвищено для медвежого ринку з високою волатильністю
        self.target_profit_pct = self.config.get('target_profit_pct', 0.06)  # Збільшено до 6% для кращого співвідношення ризик/прибуток
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.03)  # 3%
        self.max_hold_time = self.config.get('max_hold_time', 7200)  # 5 днів в хвилинах

        # Трендові індикатори
        self.trend_period = self.config.get('trend_period', 20)  # Період для тренду
        self.volume_confirmation = self.config.get('volume_confirmation', True)

        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    async def analyze_market(self, market_data: Dict[str, pd.DataFrame],
                      predictions: Dict[str, Dict]) -> Dict[str, TradeSignal]:
        """
        Аналіз ринку для свінг-трейдингу
        """
        signals = {}

        # Аналіз загального ринкового тренду (використовуємо BTC як індикатор)
        market_trend = self._analyze_market_trend(market_data)
        print(f"🌍 Swing Market Trend: {market_trend}")

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in predictions:
                continue

            df = market_data[symbol]
            pred = predictions[symbol]

            print(f"🔄 Swing analyze_market: {symbol}, data_len={len(df)}")
            signal = self._analyze_symbol(symbol, df, pred, market_trend)
            if signal:
                signals[symbol] = signal

        return signals

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame,
                       prediction: Dict, market_trend: str) -> Optional[TradeSignal]:
        """
        Аналіз окремого символу для свінг-трейдингу
        """
        if len(df) < 10:  # Зменшено до 10 для тестування
            return None

        current_price = df['close'].iloc[-1]
        predicted_change = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0)

        # Діагностичне логування
        print(f"🔍 Swing {symbol}: price={current_price:.4f}, pred_change={predicted_change:.6f}, conf={confidence:.3f}")

        # Перевірка тренду
        trend_direction, trend_strength = self._analyze_trend(df)

        # Комбінований сигнал впевненості
        combined_confidence = (confidence + trend_strength) / 2

        print(f"📈 Swing {symbol}: trend_dir={trend_direction}, trend_str={trend_strength:.3f}, combined_conf={combined_confidence:.3f}")

        if combined_confidence < self.min_confidence:
            print(f"❌ Swing {symbol}: combined_conf {combined_confidence:.3f} < min_conf {self.min_confidence}")
            return None

        # Спростимо: якщо є прогноз зміни ціни, генеруємо сигнал незалежно від тренду
        if abs(predicted_change) < 0.01:  # Мінімальний прогнозований рух 1% (було 0.1%)
            print(f"❌ Swing {symbol}: abs(pred_change) {abs(predicted_change):.6f} < 0.01")
            return None

        print(f"✅ Swing {symbol}: Генеруємо сигнал!")

        # Розрахунок волатильності для динамічного стоп-лоса
        volatility = df['close'].pct_change().std() * 100
        print(f"📊 Swing {symbol}: volatility={volatility:.2f}%")

        # Визначення напряму на основі прогнозу з динамічним стоп-лосом
        if predicted_change > 0.01:  # Позитивний прогноз (1%)
            action = TradeAction.BUY
            # Динамічний стоп-лос: 2x волатильність, але не менше 1% і не більше 4%
            dynamic_sl_pct = min(max(volatility * 2 / 100, 0.01), 0.04)
            stop_loss = current_price * (1 - dynamic_sl_pct)
            take_profit = current_price * (1 + self.target_profit_pct)
        elif predicted_change < -0.01:  # Негативний прогноз (1%)
            action = TradeAction.SELL
            # Динамічний стоп-лос: 2x волатильність, але не менше 1% і не більше 4%
            dynamic_sl_pct = min(max(volatility * 2 / 100, 0.01), 0.04)
            stop_loss = current_price * (1 + dynamic_sl_pct)
            take_profit = current_price * (1 - self.target_profit_pct)
        else:
            return None

        # Фільтр медвежого ринку - не генеруємо BUY сигнали в сильному спаді
        if market_trend == 'BEARISH' and action == TradeAction.BUY:
            print(f"🚫 Swing {symbol}: BUY сигнал заблоковано через медвежий ринок")
            return None

        # Створюємо сигнал спочатку без quantity
        signal = TradeSignal(
            action=action,
            symbol=symbol,
            confidence=combined_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=0.0,  # Тимчасово
            metadata={
                'strategy_type': 'swing_trading',
                'predicted_change': predicted_change,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'hold_period_days': self.max_hold_time / 1440,  # Конвертація в дні
                'volatility': volatility
            }
        )
        
        # Розрахунок розміру позиції через базовий метод
        signal.quantity = 0.01  # Фіксований розмір для тестування

        return signal

    def _analyze_trend(self, df: pd.DataFrame) -> tuple:
        """
        Аналіз тренду з використанням кількох індикаторів
        """
        # 1. Moving Average Trend
        short_ma = df['close'].rolling(10).mean()
        long_ma = df['close'].rolling(self.trend_period).mean()

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        ma_trend = 0
        if current_short > current_long and prev_short <= prev_long:
            ma_trend = 1  # Golden cross
        elif current_short < current_long and prev_short >= prev_long:
            ma_trend = -1  # Death cross

        # 2. Price vs MA position
        current_price = df['close'].iloc[-1]
        price_trend = 0
        if current_price > current_short > current_long:
            price_trend = 1  # Strong uptrend
        elif current_price < current_short < current_long:
            price_trend = -1  # Strong downtrend

        # 3. Momentum analysis
        momentum = df['close'].pct_change(self.trend_period).iloc[-1]
        momentum_score = 1 if momentum > 0.05 else -1 if momentum < -0.05 else 0

        # 4. Support/Resistance levels
        recent_high = df['high'].tail(self.trend_period).max()
        recent_low = df['low'].tail(self.trend_period).min()
        current_price = df['close'].iloc[-1]

        sr_score = 0
        if current_price > recent_high * 0.98:  # Near resistance
            sr_score = -0.5
        elif current_price < recent_low * 1.02:  # Near support
            sr_score = 0.5

        # Комбінований тренд
        total_score = ma_trend + price_trend + momentum_score + sr_score
        trend_strength = abs(total_score) / 4  # Нормалізація до [0, 1]

        if total_score > 1.5:
            return 'UP', min(1.0, trend_strength)
        elif total_score < -1.5:
            return 'DOWN', min(1.0, trend_strength)
        else:
            return 'SIDEWAYS', 0.5

    def _confirm_volume(self, df: pd.DataFrame) -> bool:
        """
        Підтвердження сигналу обсягом
        """
        if 'volume' not in df.columns:
            return True  # Якщо немає даних про обсяг, пропускаємо перевірку

        # Порівняння з середнім обсягом
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]

        return current_volume > avg_volume * 1.2  # Обсяг на 20% вище середнього

    def should_enter_position(self, symbol: str, signal: TradeSignal,
                            current_positions: Dict[str, Position]) -> bool:
        """
        Перевірка умов входу для свінг-трейдингу
        """
        # Перевірка кількості позицій (менше для свінг-трейдингу)
        if len(current_positions) >= min(self.max_positions, 2):
            return False

        # Перевірка, чи немає вже позиції в цьому символі
        if symbol in current_positions:
            return False

        # Висока впевненість сигналу
        if signal.confidence < self.min_confidence:
            return False

        # Додаткові перевірки для свінг-трейдингу
        trend_direction = signal.metadata.get('trend_direction')
        if trend_direction == 'SIDEWAYS':
            return False  # Не входити в бічний тренд

        return True

    def should_exit_position(self, symbol: str, position: Position,
                           current_price: float, market_data: pd.DataFrame) -> bool:
        """
        Перевірка умов виходу для свінг-трейдингу
        """
        # Перевірка часу утримання
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600  # в годинах
        max_hold_hours = self.max_hold_time / 60

        if hold_time > max_hold_hours:
            self.logger.info(f"⏰ Вихід через час утримання: {hold_time:.1f} год з {max_hold_hours}")
            return True

        # Перевірка stop-loss
        if position.stop_loss:
            if position.side == 'LONG' and current_price <= position.stop_loss:
                self.logger.info(f"🛑 Stop-loss активовано: {current_price} <= {position.stop_loss}")
                return True
            elif position.side == 'SHORT' and current_price >= position.stop_loss:
                self.logger.info(f"🛑 Stop-loss активовано: {current_price} >= {position.stop_loss}")
                return True

        # Перевірка take-profit
        if position.take_profit:
            if position.side == 'LONG' and current_price >= position.take_profit:
                self.logger.info(f"🎯 Take-profit досягнуто: {current_price} >= {position.take_profit}")
                return True
            elif position.side == 'SHORT' and current_price <= position.take_profit:
                self.logger.info(f"🎯 Take-profit досягнуто: {current_price} <= {position.take_profit}")
                return True

        # Трендовий вихід (зміна тренду)
        if market_data is not None and len(market_data) > 20:
            trend_direction, trend_strength = self._analyze_trend(market_data)

            # Якщо тренд змінився проти позиції
            if position.side == 'LONG' and trend_direction == 'DOWN' and trend_strength > 0.7:
                self.logger.info(f"📉 Вихід через зміну тренду: {trend_direction} (strength: {trend_strength:.2f})")
                return True
            elif position.side == 'SHORT' and trend_direction == 'UP' and trend_strength > 0.7:
                self.logger.info(f"📈 Вихід через зміну тренду: {trend_direction} (strength: {trend_strength:.2f})")
                return True

        # Trailing stop (динамічний стоп-лос)
        if self._should_trailing_stop(position, current_price):
            self.logger.info(f"🏃 Trailing stop активовано at {current_price}")
            return True

        return False

    def _should_trailing_stop(self, position: Position, current_price: float) -> bool:
        """
        Перевірка trailing stop
        """
        if not position.stop_loss:
            return False

        # Для довгих позицій: якщо ціна виросла на 3%, підтягуємо стоп-лос
        if position.side == 'LONG':
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct > 0.03:  # 3% прибуток
                new_stop = current_price * 0.97  # 3% від поточної ціни
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    return False  # Не виходимо, тільки оновлюємо стоп

        # Для коротких позицій: аналогічно
        elif position.side == 'SHORT':
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct > 0.03:  # 3% прибуток
                new_stop = current_price * 1.03  # 3% від поточної ціни
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    return False  # Не виходимо, тільки оновлюємо стоп

        return False

    def update_trade_stats(self, pnl: float):
        """Оновлення статистики угод"""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1

    def get_strategy_stats(self) -> Dict:
        """Статистика свінг-трейдингу"""
        stats = super().get_strategy_stats()
        stats.update({
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'avg_trade_pnl': self.total_pnl / max(self.total_trades, 1),
            'target_profit_pct': self.target_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_hold_days': self.max_hold_time / 1440,
            'trend_period': self.trend_period,
            'volume_confirmation': self.volume_confirmation
        })
        return stats

    def _analyze_market_trend(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """
        Аналіз загального тренду ринку на основі BTC
        """
        if 'BTCUSDT' not in market_data:
            return 'NEUTRAL'

        btc_df = market_data['BTCUSDT']
        if len(btc_df) < 20:
            return 'NEUTRAL'

        # Аналіз тренду за останні 20 періодів
        current_price = btc_df['close'].iloc[-1]
        price_20_periods_ago = btc_df['close'].iloc[-20]

        # Розрахунок загальної зміни
        total_change_pct = (current_price - price_20_periods_ago) / price_20_periods_ago * 100

        # Розрахунок волатильності
        volatility = btc_df['close'].pct_change().std() * 100

        # Визначення тренду
        if total_change_pct < -5:  # Спад більше 5%
            return 'BEARISH'
        elif total_change_pct > 5:  # Зростання більше 5%
            return 'BULLISH'
        else:
            return 'NEUTRAL'
    
    async def should_close_position(
        self,
        position: Position,
        current_price: float,
        market_data: pd.DataFrame
    ) -> bool:
        """
        Перевірка чи потрібно закривати позицію для свінг-трейдингу
        
        Свінг-трейдинг має більш толерантні правила:
        - Довший час утримання
        - Більші цілі по прибутку
        - Аналіз трендових розворотів
        """
        # Stop-loss і take-profit
        if position.stop_loss and current_price <= position.stop_loss:
            return True
        if position.take_profit and current_price >= position.take_profit:
            return True
        
        # Максимальний час тримання (5 днів)
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
        if hold_time > self.max_hold_time:
            return True
        
        # Аналіз трендового розвороту (якщо є достатньо даних)
        if len(market_data) >= 50:
            try:
                # Тренд розвернувся
                sma_short = market_data['close'].rolling(window=10).mean()
                sma_long = market_data['close'].rolling(window=50).mean()
                
                if position.side == 'BUY':
                    # Короткострокова MA перетнула довгострокову вниз
                    if sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
                        return True
                elif position.side == 'SELL':
                    # Короткострокова MA перетнула довгострокову вгору
                    if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
                        return True
            except Exception:
                pass
        
        return False