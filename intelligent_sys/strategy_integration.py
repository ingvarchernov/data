# -*- coding: utf-8 -*-
"""
Інтеграція торгових стратегій в intelligent_sys
Забезпечує роботу з StrategyManager та координацію стратегій
"""
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from .utils import calculate_signal_confidence

logger = logging.getLogger(__name__)


class StrategyIntegration:
    """
    Інтеграційний шар для роботи зі стратегіями
    
    Функції:
    - Ініціалізація Strategy Manager
    - Генерація торгових сигналів
    - Управління позиціями
    - Збір та агрегація метрик
    """
    
    def __init__(
        self,
        symbols: List[str],
        portfolio_value: float = 10000.0,
        strategy_config: Optional[Dict] = None
    ):
        """
        Ініціалізація інтеграції стратегій
        
        Args:
            symbols: Список торгових пар
            portfolio_value: Початковий капітал
            strategy_config: Конфігурація стратегій
        """
        self.symbols = symbols
        self.portfolio_value = portfolio_value
        self.strategy_config = strategy_config or {}
        self.strategy_manager = None
        self.initialized = False
        
        logger.info(f"🎯 StrategyIntegration створено для {len(symbols)} символів")
    
    def initialize(self) -> bool:
        """
        Ініціалізація Strategy Manager
        
        Returns:
            bool: True якщо успішно
        """
        try:
            # Додаємо батьківську директорію до sys.path для імпорту strategy_manager
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Імпорт стратегій напряму
            from strategies.scalping import ScalpingStrategy
            from strategies.day_trading import DayTradingStrategy
            from strategies.swing_trading import SwingTradingStrategy
            
            # Ініціалізація активних стратегій
            self.active_strategies = []
            
            if self.strategy_config.get('enable_scalping', False):
                self.active_strategies.append(ScalpingStrategy(self.symbols))
                logger.info("✅ Scalping strategy enabled")
                
            if self.strategy_config.get('enable_day_trading', True):
                self.active_strategies.append(DayTradingStrategy(self.symbols))
                logger.info("✅ Day trading strategy enabled")
                
            if self.strategy_config.get('enable_swing_trading', True):
                self.active_strategies.append(SwingTradingStrategy(self.symbols))
                logger.info("✅ Swing trading strategy enabled")
            
            self.strategy_manager = None  # Не використовуємо manager
            
            self.initialized = True
            logger.info(f"✅ Strategies initialized: {len(self.active_strategies)} active")
            return True
            
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації Strategy Manager: {e}", exc_info=True)
            return False
    
    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        predictions: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Генерація торгових сигналів
        
        Args:
            market_data: Історичні дані по символах
            predictions: ML прогнози (опціонально)
            
        Returns:
            Dict з торговими сигналами
        """
        if not self.initialized or not self.active_strategies:
            logger.warning("⚠️ Strategies не ініціалізовано")
            return {}
        
        try:
            # Якщо немає прогнозів, створюємо заглушки
            if predictions is None:
                predictions = self._create_dummy_predictions(market_data)
            
            # Генерація сигналів з усіх активних стратегій
            all_signals = {}
            for strategy in self.active_strategies:
                try:
                    # Перевіряємо чи метод async
                    if hasattr(strategy.analyze_market, '__call__'):
                        import inspect
                        if inspect.iscoroutinefunction(strategy.analyze_market):
                            strategy_signals = await strategy.analyze_market(market_data, predictions)
                        else:
                            strategy_signals = strategy.analyze_market(market_data, predictions)
                    else:
                        strategy_signals = strategy.analyze_market(market_data, predictions)
                    all_signals.update(strategy_signals)
                except Exception as e:
                    logger.error(f"❌ Помилка в {strategy.name}: {e}")
            
            if all_signals:
                logger.info(f"📊 Згенеровано {len(all_signals)} торгових сигналів")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"❌ Помилка генерації сигналів: {e}", exc_info=True)
            return {}
    
    def _create_dummy_predictions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Створення базових прогнозів на основі трендів
        Використовується, якщо немає ML моделей
        """
        predictions = {}
        
        for symbol, df in market_data.items():
            if df.empty or len(df) < 2:
                continue
            
            try:
                current_price = df['close'].iloc[-1]
                price_change = df['close'].pct_change().iloc[-1]
                
                # Простий прогноз на основі моментуму
                predicted_change = price_change * 1.1
                predicted_price = current_price * (1 + predicted_change)
                confidence = calculate_signal_confidence(predicted_change, df)
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predicted_change': predicted_change,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"❌ Помилка створення прогнозу для {symbol}: {e}")
        
        return predictions
    
    async def check_close_positions(
        self,
        current_prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, bool]:
        """
        Перевірка необхідності закриття позицій
        
        Args:
            current_prices: Поточні ціни символів
            market_data: Ринкові дані для аналізу
            
        Returns:
            Dict[symbol, should_close]: Рішення про закриття
        """
        if not self.initialized or not self.active_strategies:
            logger.warning("⚠️ Strategies не ініціалізовано")
            return {}
        
        try:
            # Перевірка для кожної стратегії
            close_decisions = {}
            for strategy in self.active_strategies:
                try:
                    # Перевіряємо позиції стратегії
                    for symbol in strategy.symbols:
                        if symbol in strategy.positions and strategy.positions[symbol]:
                            position = strategy.positions[symbol]
                            current_price = current_prices.get(symbol, 0)
                            if current_price and strategy._should_close_position(position, current_price):
                                close_decisions[symbol] = True
                except Exception as e:
                    logger.error(f"❌ Помилка перевірки {strategy.name}: {e}")
            
            if close_decisions:
                logger.info(f"🔔 Рекомендовано закрити {len(close_decisions)} позицій")
            
            return close_decisions
            
        except Exception as e:
            logger.error(f"❌ Помилка перевірки закриття позицій: {e}", exc_info=True)
            return {}
    
    def validate_signal(self, signal: Any) -> tuple:
        """
        Валідація торгового сигналу
        
        Returns:
            (is_valid, reason)
        """
        if not self.initialized or not self.active_strategies:
            return False, "Strategies не ініціалізовано"
        
        try:
            # Базова валідація
            if not hasattr(signal, 'action') or not hasattr(signal, 'confidence'):
                return False, "Невалідна структура сигналу"
            
            if signal.confidence < 0.05:
                return False, f"Занадто низька впевненість: {signal.confidence}"
            
            return True, "OK"
        except Exception as e:
            logger.error(f"❌ Помилка валідації сигналу: {e}")
            return False, str(e)
    
    def calculate_position_size(self, signal: Any) -> float:
        """
        Розрахунок розміру позиції
        
        Returns:
            float: Розмір позиції
        """
        if not self.initialized or not self.active_strategies:
            logger.warning("⚠️ Strategies не ініціалізовано")
            return 0.0
        
        try:
            # Базовий розрахунок: 2% від портфеля з урахуванням впевненості
            base_risk = self.portfolio_value * 0.02
            position_size = base_risk * signal.confidence
            return position_size
        except Exception as e:
            logger.error(f"❌ Помилка розрахунку розміру позиції: {e}")
            return 0.0
    
    def record_trade(self, symbol: str, pnl: float, strategy_name: str = None):
        """
        Запис результату угоди
        
        Args:
            symbol: Торгова пара
            pnl: Прибуток/збиток
            strategy_name: Назва стратегії (опціонально)
        """
        if not self.initialized or not self.active_strategies:
            logger.warning("⚠️ Strategies не ініціалізовано")
            return
        
        try:
            # Записуємо в відповідну стратегію
            for strategy in self.active_strategies:
                if strategy_name and strategy.name == strategy_name:
                    strategy.record_trade(symbol, pnl)
                    break
            logger.debug(f"📝 Записано угоду {symbol}: PnL=${pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Помилка запису угоди: {e}")
    
    def update_portfolio_value(self, new_value: float):
        """
        Оновлення вартості портфеля
        
        Args:
            new_value: Нова вартість портфеля
        """
        if not self.initialized or not self.active_strategies:
            return
        
        try:
            self.portfolio_value = new_value
            logger.debug(f"💰 Портфель оновлено: ${new_value:.2f}")
        except Exception as e:
            logger.error(f"❌ Помилка оновлення портфеля: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Отримання статистики продуктивності
        
        Returns:
            Dict з метриками продуктивності
        """
        if not self.initialized or not self.active_strategies:
            return {
                'initialized': False,
                'portfolio_value': self.portfolio_value,
                'total_trades': 0,
                'win_rate': 0.0
            }
        
        try:
            # Збираємо статистику з усіх стратегій
            summary = {
                'portfolio_value': self.portfolio_value,
                'strategies': {}
            }
            
            total_trades = 0
            total_winning = 0
            
            for strategy in self.active_strategies:
                strategy_stats = {
                    'name': strategy.name,
                    'total_trades': getattr(strategy, 'total_trades', 0),
                    'winning_trades': getattr(strategy, 'winning_trades', 0),
                    'total_pnl': getattr(strategy, 'total_pnl', 0.0)
                }
                summary['strategies'][strategy.name] = strategy_stats
                total_trades += strategy_stats['total_trades']
                total_winning += strategy_stats['winning_trades']
            
            summary['total_trades'] = total_trades
            summary['win_rate'] = (total_winning / total_trades * 100) if total_trades > 0 else 0.0
            
            return summary
        except Exception as e:
            logger.error(f"❌ Помилка отримання статистики: {e}")
            return {'error': str(e)}
    
    def reset_daily_stats(self):
        """Скидання денної статистики"""
        if not self.initialized or not self.active_strategies:
            return
        
        try:
            for strategy in self.active_strategies:
                if hasattr(strategy, 'reset_daily_stats'):
                    strategy.reset_daily_stats()
            logger.info("🔄 Денна статистика стратегій скинута")
        except Exception as e:
            logger.error(f"❌ Помилка скидання статистики: {e}")
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """
        Отримання всіх активних позицій
        
        Returns:
            Dict з активними позиціями по всіх стратегіях
        """
        if not self.initialized or not self.active_strategies:
            return {}
        
        try:
            all_positions = {}
            for strategy in self.active_strategies:
                for symbol, position in strategy.positions.items():
                    if position:  # Якщо позиція активна
                        all_positions[f"{strategy.name}_{symbol}"] = {
                            'strategy': strategy.name,
                            'symbol': symbol,
                            'position': position
                        }
            return all_positions
        except Exception as e:
            logger.error(f"❌ Помилка отримання позицій: {e}")
            return {}
    
    def shutdown(self):
        """Коректне завершення роботи"""
        if self.active_strategies:
            logger.info("🔄 Завершення роботи Strategy Integration...")
            # Очищаємо стратегії
            self.active_strategies.clear()
        
        self.initialized = False
        logger.info("✅ Strategy Integration завершено")


# Допоміжна функція для створення з налаштуваннями за замовчуванням
def create_strategy_integration(
    symbols: List[str],
    portfolio_value: float = 10000.0,
    enable_scalping: bool = False,
    enable_day_trading: bool = True,
    enable_swing_trading: bool = True
) -> StrategyIntegration:
    """
    Швидке створення StrategyIntegration з типовими налаштуваннями
    
    Args:
        symbols: Список торгових пар
        portfolio_value: Початковий капітал
        enable_scalping: Увімкнути скальпінг
        enable_day_trading: Увімкнути денну торгівлю
        enable_swing_trading: Увімкнути свінг-трейдинг
        
    Returns:
        StrategyIntegration: Готовий до використання об'єкт
    """
    config = {
        'enable_scalping': enable_scalping,
        'enable_day_trading': enable_day_trading,
        'enable_swing_trading': enable_swing_trading,
        'risk_config': {
            'max_risk_per_trade': 0.02,  # 2% на угоду
            'max_daily_loss': 0.05,  # 5% максимальний денний збиток
            'max_positions': 10  # Максимум 10 одночасних позицій
        }
    }
    
    integration = StrategyIntegration(
        symbols=symbols,
        portfolio_value=portfolio_value,
        strategy_config=config
    )
    
    # Ініціалізація
    if integration.initialize():
        logger.info("✅ StrategyIntegration створено та ініціалізовано")
    else:
        logger.error("❌ Помилка ініціалізації StrategyIntegration")
    
    return integration
