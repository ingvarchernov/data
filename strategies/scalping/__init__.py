"""
Скальпінгова стратегія - короткострокові угоди на малих рухах ціни
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from strategies.base import TradingStrategy, TradeSignal, TradeAction, Position


class ScalpingStrategy(TradingStrategy):
    """
    Скальпінгова стратегія для короткострокової торгівлі

    Особливості:
    - Тривалість позицій: 5-60 хвилин
    - Цілі: 0.5-2% прибуток на угоду
    - Stop-loss: 0.3-1%
    - Висока частота угод
    """

    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        super().__init__("scalping", symbols, config)

        # Параметри скальпінгу
        self.min_confidence = 0.08  # Підвищено для медвежого ринку з високою волатильністю
        self.target_profit_pct = 0.03
        self.stop_loss_pct = 0.015
        self.max_hold_time = self.config.get('max_hold_time', 45)  # 45 хвилин
        self.min_volume_threshold = self.config.get('min_volume_threshold', 100000)  # Зменшено до $100K

        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def analyze_market(self, market_data: Dict[str, pd.DataFrame],
                      predictions: Dict[str, Dict]) -> Dict[str, TradeSignal]:
        """
        Аналіз ринку для скальпінгу
        """
        signals = {}

        # Аналіз загального ринкового тренду (використовуємо BTC як індикатор)
        market_trend = self._analyze_market_trend(market_data)
        print(f"🌍 Scalp Market Trend: {market_trend}")

        for symbol in self.symbols:
            if symbol not in market_data or symbol not in predictions:
                continue

            df = market_data[symbol]
            pred = predictions[symbol]

            signal = self._analyze_symbol(symbol, df, pred, market_trend)
            if signal:
                signals[symbol] = signal

        return signals

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame,
                       prediction: Dict, market_trend: str) -> Optional[TradeSignal]:
        """
        Аналіз окремого символу
        """
        if len(df) < 10:  # Потрібно мінімум 10 свічок
            return None

        current_price = df['close'].iloc[-1]
        predicted_change = prediction.get('change_percent', 0)
        confidence = prediction.get('confidence', 0)

        # Діагностичне логування
        print(f"🔍 Scalp {symbol}: price={current_price:.4f}, pred_change={predicted_change:.6f}, conf={confidence:.3f}")

        # Перевірка мінімальної впевненості
        if confidence < self.min_confidence:
            print(f"❌ Scalp {symbol}: conf {confidence:.3f} < min_conf {self.min_confidence}")
            return None

        # Перевірка мінімального обсягу
        avg_volume = df['volume'].tail(5).mean() * current_price
        print(f"💰 Scalp {symbol}: avg_volume={avg_volume:.0f}, min_threshold={self.min_volume_threshold}")
        if avg_volume < self.min_volume_threshold:
            print(f"❌ Scalp {symbol}: volume too low")
            return None

        # Аналіз волатильності (не надто висока для скальпінгу)
        volatility = df['close'].pct_change().std() * 100
        print(f"📊 Scalp {symbol}: volatility={volatility:.2f}%")
        if volatility > 5:  # Занадто волатильно для скальпінгу
            print(f"❌ Scalp {symbol}: volatility too high")
            return None

        # Визначення напряму з динамічним стоп-лосом на основі волатильності
        if predicted_change > 0.001:  # Мінімальний прогнозований рух 0.1%
            action = TradeAction.BUY
            # Динамічний стоп-лос: 1.5x волатильність, але не менше 0.5% і не більше 2%
            dynamic_sl_pct = min(max(volatility * 1.5 / 100, 0.005), 0.02)
            stop_loss = current_price * (1 - dynamic_sl_pct)
            take_profit = current_price * (1 + self.target_profit_pct)
        elif predicted_change < -0.001:  # Мінімальний прогнозований рух -0.1%
            action = TradeAction.SELL
            # Динамічний стоп-лос: 1.5x волатильність, але не менше 0.5% і не більше 2%
            dynamic_sl_pct = min(max(volatility * 1.5 / 100, 0.005), 0.02)
            stop_loss = current_price * (1 + dynamic_sl_pct)
            take_profit = current_price * (1 - self.target_profit_pct)
        else:
            print(f"❌ Scalp {symbol}: pred_change {predicted_change:.6f} not > 0.001 or < -0.001")
            return None

        print(f"✅ Scalp {symbol}: Генеруємо сигнал!")

        # Фільтр медвежого ринку - не генеруємо BUY сигнали в сильному спаді
        if market_trend == 'BEARISH' and action == TradeAction.BUY:
            print(f"🚫 Scalp {symbol}: BUY сигнал заблоковано через медвежий ринок")
            return None

        # Розрахунок розміру позиції - фіксована сума $1000
        invest_amount = 1000.0  # Фіксована сума для кожної торгівлі
        quantity = invest_amount / current_price

        return TradeSignal(
            action=action,
            symbol=symbol,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            metadata={
                'strategy_type': 'scalping',
                'predicted_change': predicted_change,
                'volatility': volatility,
                'avg_volume': avg_volume
            }
        )

    def should_enter_position(self, symbol: str, signal: TradeSignal,
                            current_positions: Dict[str, Position]) -> bool:
        """
        Перевірка умов входу для скальпінгу
        """
        # Перевірка кількості позицій
        if len(current_positions) >= self.max_positions:
            return False

        # Перевірка, чи немає вже позиції в цьому символі
        if symbol in current_positions:
            return False

        # Додаткові перевірки для скальпінгу
        if signal.confidence < self.min_confidence:
            return False

        return True

    def should_exit_position(self, symbol: str, position: Position,
                           current_price: float, market_data: pd.DataFrame) -> bool:
        """
        Перевірка умов виходу для скальпінгу
        """
        # Перевірка часу утримання
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 60  # в хвилинах
        if hold_time > self.max_hold_time:
            self.logger.info(f"⏰ Вихід через час утримання: {hold_time:.1f} хв")
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

        # Динамічний вихід при зміні тренду (якщо є нові дані)
        if market_data is not None and len(market_data) > 5:
            recent_trend = market_data['close'].tail(3).pct_change().mean()
            if position.side == 'LONG' and recent_trend < -0.002:  # Тренд змінився вниз
                self.logger.info(f"📉 Вихід через зміну тренду: {recent_trend:.4f}")
                return True
            elif position.side == 'SHORT' and recent_trend > 0.002:  # Тренд змінився вгору
                self.logger.info(f"📈 Вихід через зміну тренду: {recent_trend:.4f}")
                return True

        return False

    def update_trade_stats(self, pnl: float):
        """Оновлення статистики угод"""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1

    def get_strategy_stats(self) -> Dict:
        """Статистика скальпінгової стратегії"""
        stats = super().get_strategy_stats()
        stats.update({
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'avg_trade_pnl': self.total_pnl / max(self.total_trades, 1),
            'target_profit_pct': self.target_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_hold_time': self.max_hold_time
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
        Перевірка чи потрібно закривати позицію для скальпінгу
        
        Скальпінг має дуже жорсткі правила закриття:
        - Швидке закриття при досягненні цілі або стоп-лосс
        - Обмежений час тримання позиції
        """
        # Stop-loss і take-profit (пріоритет!)
        if position.stop_loss and current_price <= position.stop_loss:
            return True
        if position.take_profit and current_price >= position.take_profit:
            return True
        
        # Максимальний час тримання (критично для скальпінгу)
        hold_time = (datetime.now() - position.entry_time).total_seconds() / 60
        if hold_time > self.max_hold_time:
            return True
        
        # Швидке закриття при маленькому прибутку (якщо час спливає)
        if hold_time > self.max_hold_time * 0.7:  # 70% часу минуло
            current_pnl_pct = (current_price - position.entry_price) / position.entry_price
            if position.side == 'BUY' and current_pnl_pct > 0.003:  # 0.3% прибуток
                return True
            elif position.side == 'SELL' and current_pnl_pct < -0.003:
                return True
        
        return False