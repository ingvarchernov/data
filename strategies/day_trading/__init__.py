"""
Стратегія денної торгівлі - позиції протягом торгового дня
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, time

from strategies.base import TradingStrategy, TradeSignal, TradeAction, Position


class DayTradingStrategy(TradingStrategy):
    """
    Стратегія денної торгівлі

    Особливості:
    - Тривалість позицій: 2-8 годин
    - Цілі: 2-5% прибуток на угоду
    - Використання технічних індикаторів
    - Фокус на інтрадей трендах
    """

    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        super().__init__("day_trading", symbols, config)

        # Параметри денної торгівлі
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.target_profit_pct = self.config.get('target_profit_pct', 0.03)  # 3%
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.015)  # 1.5%
        self.max_hold_time = self.config.get('max_hold_time', 480)  # 8 годин

        # Часові рамки для торгівлі
        self.trading_start = time(9, 0)  # 9:00 UTC
        self.trading_end = time(16, 0)   # 16:00 UTC

        # Технічні індикатори
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.5)

        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def analyze_market(self, market_data: Dict[str, pd.DataFrame],
                      predictions: Dict[str, Dict]) -> Dict[str, TradeSignal]:
        """
        Аналіз ринку для денної торгівлі
        """
        signals = {}

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in predictions:
                continue

            df = market_data[symbol]
            pred = predictions[symbol]

            signal = self._analyze_symbol(symbol, df, pred)
            if signal:
                signals[symbol] = signal

        return signals

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame,
                       prediction: Dict) -> Optional[TradeSignal]:
        """
        Аналіз окремого символу з технічними індикаторами
        """
        if len(df) < 50:  # Потрібно більше даних для технічних індикаторів
            return None

        current_price = df['close'].iloc[-1]
        predicted_change = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0)

        # Перевірка часу торгівлі
        current_time = datetime.now().time()
        if not (self.trading_start <= current_time <= self.trading_end):
            return None

        # Комплексний аналіз
        technical_score = self._calculate_technical_score(df)

        # Комбінований сигнал
        combined_confidence = (confidence + technical_score) / 2

        if combined_confidence < self.min_confidence:
            return None

        # Визначення напряму на основі прогнозу та технічних індикаторів
        if predicted_change > 0.02 and technical_score > 0.6:  # Сильний сигнал вгору
            action = TradeAction.BUY
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.target_profit_pct)
        elif predicted_change < -0.02 and technical_score < 0.4:  # Сильний сигнал вниз
            action = TradeAction.SELL
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.target_profit_pct)
        else:
            return None

        # Розрахунок розміру позиції з урахуванням волатильності
        volatility = df['close'].pct_change().std()
        adjusted_risk = self.risk_per_trade * (1 + volatility * 2)  # Збільшуємо ризик при високій волатильності

        quantity = self.calculate_position_size(10000, current_price, stop_loss)

        return TradeSignal(
            action=action,
            symbol=symbol,
            confidence=combined_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            metadata={
                'strategy_type': 'day_trading',
                'predicted_change': predicted_change,
                'technical_score': technical_score,
                'volatility': volatility,
                'trading_time': current_time.isoformat()
            }
        )

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Розрахунок технічного скорингу
        """
        score = 0.5  # Базовий score

        # RSI аналіз
        if 'RSI' in df.columns:
            current_rsi = df['RSI'].iloc[-1]
            if current_rsi < self.rsi_oversold:
                score += 0.2  # Oversold - сигнал на покупку
            elif current_rsi > self.rsi_overbought:
                score -= 0.2  # Overbought - сигнал на продаж

        # Тренд аналіз (moving averages)
        if 'EMA' in df.columns and len(df) > 20:
            short_ma = df['close'].rolling(10).mean().iloc[-1]
            long_ma = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]

            if current_price > short_ma > long_ma:
                score += 0.15  # Сильний висхідний тренд
            elif current_price < short_ma < long_ma:
                score -= 0.15  # Сильний низхідний тренд

        # Обсяг аналіз
        if len(df) > 10:
            avg_volume = df['volume'].rolling(10).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]

            if current_volume > avg_volume * self.volume_multiplier:
                score += 0.1  # Високий обсяг підтверджує сигнал

        # Волатильність (Bollinger Bands)
        if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
            upper = df['Upper_Band'].iloc[-1]
            lower = df['Lower_Band'].iloc[-1]
            current_price = df['close'].iloc[-1]

            # Якщо ціна близько до нижньої смуги - потенціал росту
            if current_price < lower * 1.02:
                score += 0.1
            # Якщо ціна близько до верхньої смуги - потенціал падіння
            elif current_price > upper * 0.98:
                score -= 0.1

        return max(0, min(1, score))  # Нормалізація до [0, 1]

    def should_enter_position(self, symbol: str, signal: TradeSignal,
                            current_positions: Dict[str, Position]) -> bool:
        """
        Перевірка умов входу для денної торгівлі
        """
        # Перевірка кількості позицій
        if len(current_positions) >= self.max_positions:
            return False

        # Перевірка, чи немає вже позиції в цьому символі
        if symbol in current_positions:
            return False

        # Денні обмеження
        current_time = datetime.now().time()
        if not (self.trading_start <= current_time <= self.trading_end):
            return False

        # Мінімальна впевненість сигналу
        if signal.confidence < self.min_confidence:
            return False

        return True

    def should_exit_position(self, symbol: str, position: Position,
                           current_price: float, market_data: pd.DataFrame) -> bool:
        """
        Перевірка умов виходу для денної торгівлі
        """
        # Перевірка часу утримання
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600  # в годинах
        if hold_time > self.max_hold_time / 60:  # Конвертація в години
            self.logger.info(f"⏰ Вихід через час утримання: {hold_time:.1f} год")
            return True

        # Перевірка кінця торгового дня
        current_time = datetime.now().time()
        if current_time >= time(15, 30):  # За 30 хв до кінця
            self.logger.info(f"🏁 Вихід через кінець торгового дня: {current_time}")
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

        # Технічний вихід (зміна тренду)
        if market_data is not None and len(market_data) > 10:
            recent_prices = market_data['close'].tail(5)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

            # Якщо тренд змінився проти позиції
            if position.side == 'LONG' and trend < -0.005:  # -0.5%
                self.logger.info(f"📉 Вихід через зміну тренду: {trend:.4f}")
                return True
            elif position.side == 'SHORT' and trend > 0.005:  # +0.5%
                self.logger.info(f"📈 Вихід через зміну тренду: {trend:.4f}")
                return True

        return False

    def update_trade_stats(self, pnl: float):
        """Оновлення статистики угод"""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1

    def get_strategy_stats(self) -> Dict:
        """Статистика денної торгівлі"""
        stats = super().get_strategy_stats()
        stats.update({
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'avg_trade_pnl': self.total_pnl / max(self.total_trades, 1),
            'target_profit_pct': self.target_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_hold_time': self.max_hold_time,
            'trading_hours': f"{self.trading_start}-{self.trading_end}"
        })
        return stats