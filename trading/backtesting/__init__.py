"""
Система бектестингу для тестування торгових стратегій
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from strategies.base import TradingStrategy, TradeSignal, TradeAction, Position

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Результати бектестингу"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime

    # Фінансові метрики
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

    # Ризикові метрики
    volatility: float
    max_consecutive_losses: int
    avg_trade_duration: float

    # Детальна інформація
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    drawdown_curve: List[float]


class Backtester:
    """
    Система бектестингу для тестування стратегій
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% комісія
        self.slippage = 0.0005  # 0.05% slippage

    def run_backtest(self,
                    strategy: TradingStrategy,
                    market_data: Dict[str, pd.DataFrame],
                    predictions: Dict[str, Dict],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict[str, BacktestResult]:
        """
        Запуск бектестингу для стратегії
        """
        results = {}

        for symbol in strategy.symbols:
            if symbol not in market_data:
                logger.warning(f"Немає даних для символу {symbol}")
                continue

            logger.info(f"🧪 Тестування стратегії {strategy.name} на {symbol}")

            result = self._backtest_symbol(
                strategy, symbol, market_data[symbol],
                predictions.get(symbol, {}), start_date, end_date
            )

            if result:
                results[symbol] = result

        return results

    def _backtest_symbol(self,
                        strategy: TradingStrategy,
                        symbol: str,
                        df: pd.DataFrame,
                        predictions: Dict,
                        start_date: Optional[datetime],
                        end_date: Optional[datetime]) -> Optional[BacktestResult]:
        """
        Бектестинг для окремого символу
        """
        # Фільтрація даних по даті
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if len(df) < 100:
            logger.warning(f"Недостатньо даних для {symbol}: {len(df)} записів")
            return None

        # Ініціалізація
        capital = self.initial_capital
        positions = {}
        trades = []
        equity_curve = [capital]
        drawdown_curve = [0.0]

        peak_capital = capital
        max_drawdown = 0.0

        # Ітерація по кожній свічці
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_time = current_bar.name
            current_price = current_bar['close']

            # Генерація прогнозу для поточного моменту (якщо є)
            current_predictions = self._get_prediction_at_time(predictions, current_time)

            # Аналіз ринку стратегією
            market_snapshot = {symbol: df.iloc[:i+1]}  # Дані до поточного моменту
            signals = strategy.analyze_market(market_snapshot, {symbol: current_predictions})

            # Обробка сигналів
            if symbol in signals:
                signal = signals[symbol]

                # Вхід в позицію
                if signal.action in [TradeAction.BUY, TradeAction.SELL] and symbol not in positions:
                    if self._can_enter_position(strategy, capital, signal):
                        position = self._create_backtest_position(signal, current_price, current_time)
                        positions[symbol] = position

                        # Комісія за відкриття
                        commission = capital * self.commission
                        capital -= commission

                        logger.debug(f"📈 Відкрита позиція {symbol} at {current_price}")

                # Вихід з позиції
                elif signal.action == TradeAction.CLOSE and symbol in positions:
                    position = positions[symbol]
                    pnl = self._calculate_pnl(position, current_price)

                    # Комісія за закриття
                    commission = abs(pnl) * self.commission
                    capital += pnl - commission

                    # Запис угоди
                    trade = {
                        'symbol': symbol,
                        'entry_time': position.entry_time,
                        'exit_time': current_time,
                        'entry_price': position.entry_price,
                        'exit_price': current_price,
                        'quantity': position.quantity,
                        'pnl': pnl,
                        'commission': commission,
                        'duration': (current_time - position.entry_time).total_seconds() / 3600,  # години
                        'side': position.side
                    }
                    trades.append(trade)

                    del positions[symbol]
                    logger.debug(f"📉 Закрита позиція {symbol} at {current_price}, P&L: {pnl:.2f}")

            # Оновлення equity curve
            unrealized_pnl = sum([
                self._calculate_pnl(pos, current_price)
                for pos in positions.values()
            ])

            total_equity = capital + unrealized_pnl
            equity_curve.append(total_equity)

            # Розрахунок drawdown
            peak_capital = max(peak_capital, total_equity)
            current_drawdown = (peak_capital - total_equity) / peak_capital
            max_drawdown = max(max_drawdown, current_drawdown)
            drawdown_curve.append(current_drawdown)

        # Розрахунок фінальних метрик
        if not trades:
            logger.warning(f"Жодної угоди не було здійснено для {symbol}")
            return None

        return self._calculate_backtest_metrics(
            strategy.name, symbol, trades, equity_curve, drawdown_curve,
            df.index[0], df.index[-1]
        )

    def _get_prediction_at_time(self, predictions: Dict, timestamp: datetime) -> Dict:
        """Отримання прогнозу на конкретний час"""
        # Спрощена версія - повертаємо статичний прогноз
        return predictions

    def _can_enter_position(self, strategy: TradingStrategy, capital: float, signal: TradeSignal) -> bool:
        """Перевірка можливості входу в позицію"""
        if not signal.quantity or signal.quantity <= 0:
            return False

        position_value = signal.entry_price * signal.quantity
        max_position_size = capital * 0.1  # Максимум 10% від капіталу

        return position_value <= max_position_size

    def _create_backtest_position(self, signal: TradeSignal, current_price: float, timestamp: datetime) -> Position:
        """Створення позиції для бектестингу"""
        from strategies.base import Position

        side = 'LONG' if signal.action == TradeAction.BUY else 'SHORT'

        return Position(
            symbol=signal.symbol,
            side=side,
            entry_price=signal.entry_price or current_price,
            quantity=signal.quantity or 0.001,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=timestamp
        )

    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Розрахунок P&L позиції"""
        if position.side == 'LONG':
            return (current_price - position.entry_price) * position.quantity
        else:  # SHORT
            return (position.entry_price - current_price) * position.quantity

    def _calculate_backtest_metrics(self,
                                   strategy_name: str,
                                   symbol: str,
                                   trades: List[Dict],
                                   equity_curve: List[float],
                                   drawdown_curve: List[float],
                                   start_date: datetime,
                                   end_date: datetime) -> BacktestResult:
        """Розрахунок метрик бектестингу"""

        # Базові метрики
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades

        if total_trades == 0:
            return None

        # Фінансові метрики
        total_pnl = sum(t['pnl'] for t in trades)
        total_return = total_pnl / self.initial_capital

        win_rate = winning_trades / total_trades

        winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]

        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0

        profit_factor = (sum(winning_pnls) / abs(sum(losing_pnls))) if losing_pnls else float('inf')

        # Ризикові метрики
        returns = np.diff(equity_curve) / equity_curve[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Максимальна серія програшів
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade['pnl'] < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Середня тривалість угоди
        avg_trade_duration = np.mean([t['duration'] for t in trades])

        # Максимальний drawdown
        max_drawdown = max(drawdown_curve)

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            max_consecutive_losses=max_consecutive_losses,
            avg_trade_duration=avg_trade_duration,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve
        )

    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Візуалізація результатів бектестингу"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results: {result.strategy_name} - {result.symbol}')

        # Equity curve
        axes[0, 0].plot(result.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trades')
        axes[0, 0].set_ylabel('Capital')
        axes[0, 0].grid(True)

        # Drawdown curve
        axes[0, 1].plot(result.drawdown_curve, color='red')
        axes[0, 1].set_title('Drawdown Curve')
        axes[0, 1].set_xlabel('Trades')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True)

        # P&L distribution
        pnls = [t['pnl'] for t in result.trades]
        axes[1, 0].hist(pnls, bins=50, alpha=0.7)
        axes[1, 0].set_title('P&L Distribution')
        axes[1, 0].set_xlabel('P&L')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].grid(True)

        # Trade duration vs P&L
        durations = [t['duration'] for t in result.trades]
        axes[1, 1].scatter(durations, pnls, alpha=0.6)
        axes[1, 1].set_title('Trade Duration vs P&L')
        axes[1, 1].set_xlabel('Duration (hours)')
        axes[1, 1].set_ylabel('P&L')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Графік збережено: {save_path}")
        else:
            plt.show()

    def print_summary(self, results: Dict[str, BacktestResult]):
        """Вивід зведення результатів"""
        print("\n" + "="*80)
        print("📊 ЗВЕДЕННЯ РЕЗУЛЬТАТІВ БЕКТЕСТИНГУ")
        print("="*80)

        for symbol, result in results.items():
            print(f"\n🪙 Символ: {symbol}")
            print(f"📈 Стратегія: {result.strategy_name}")
            print(f"📅 Період: {result.start_date.date()} - {result.end_date.date()}")

            print(f"\n💰 Фінансові результати:")
            print(f"  Загальний прибуток: {result.total_return:.2%}")
            print(f"  Загальна кількість угод: {result.total_trades}")
            print(f"  Прибуткових угод: {result.winning_trades}")
            print(f"  Відсоток перемог: {result.win_rate:.1%}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")

            print(f"\n⚠️ Ризикові метрики:")
            print(f"  Максимальний drawdown: {result.max_drawdown:.1%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Волатильність: {result.volatility:.1%}")
            print(f"  Макс. серія програшів: {result.max_consecutive_losses}")

            print(f"\n⏱️ Часові метрики:")
            print(f"  Середня тривалість угоди: {result.avg_trade_duration:.1f} годин")
            print(f"  Середній прибуток: ${result.avg_win:.2f}")
            print(f"  Середній збиток: ${result.avg_loss:.2f}")

        print("\n" + "="*80)


# Функція для паралельного тестування
def run_parallel_backtest(strategies: List[TradingStrategy],
                         market_data: Dict[str, pd.DataFrame],
                         predictions: Dict[str, Dict],
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         max_workers: int = 4) -> Dict[str, Dict[str, BacktestResult]]:
    """
    Паралельний запуск бектестингу для кількох стратегій
    """
    backtester = Backtester()
    all_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for strategy in strategies:
            future = executor.submit(
                backtester.run_backtest,
                strategy, market_data, predictions, start_date, end_date
            )
            futures[future] = strategy.name

        for future in as_completed(futures):
            strategy_name = futures[future]
            try:
                results = future.result()
                all_results[strategy_name] = results
                logger.info(f"✅ Завершено тестування стратегії {strategy_name}")
            except Exception as e:
                logger.error(f"❌ Помилка тестування стратегії {strategy_name}: {e}")

    return all_results