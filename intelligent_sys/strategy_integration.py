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
            
            # Імпорт StrategyManager
            from strategy_manager import StrategyManager
            
            # Створення Strategy Manager
            self.strategy_manager = StrategyManager(
                symbols=self.symbols,
                portfolio_value=self.portfolio_value,
                enable_scalping=self.strategy_config.get('enable_scalping', False),
                enable_day_trading=self.strategy_config.get('enable_day_trading', True),
                enable_swing_trading=self.strategy_config.get('enable_swing_trading', True),
                risk_config=self.strategy_config.get('risk_config', {})
            )
            
            self.initialized = True
            logger.info("✅ Strategy Manager ініціалізовано успішно")
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
        if not self.initialized or not self.strategy_manager:
            logger.warning("⚠️ Strategy Manager не ініціалізовано")
            return {}
        
        try:
            # Якщо немає прогнозів, створюємо заглушки
            if predictions is None:
                predictions = self._create_dummy_predictions(market_data)
            
            # Генерація сигналів через Strategy Manager
            signals = await self.strategy_manager.analyze_and_generate_signals(
                market_data=market_data,
                predictions=predictions
            )
            
            if signals:
                logger.info(f"📊 Згенеровано {len(signals)} торгових сигналів")
            
            return signals
            
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
        if not self.initialized or not self.strategy_manager:
            logger.warning("⚠️ Strategy Manager не ініціалізовано")
            return {}
        
        try:
            close_decisions = await self.strategy_manager.should_close_positions(
                current_prices=current_prices,
                market_data=market_data
            )
            
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
        if not self.initialized or not self.strategy_manager:
            return False, "Strategy Manager не ініціалізовано"
        
        try:
            return self.strategy_manager.validate_signal(signal)
        except Exception as e:
            logger.error(f"❌ Помилка валідації сигналу: {e}")
            return False, str(e)
    
    def calculate_position_size(self, signal: Any) -> float:
        """
        Розрахунок розміру позиції
        
        Returns:
            float: Розмір позиції
        """
        if not self.initialized or not self.strategy_manager:
            logger.warning("⚠️ Strategy Manager не ініціалізовано")
            return 0.0
        
        try:
            return self.strategy_manager.calculate_position_size(signal)
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
        if not self.initialized or not self.strategy_manager:
            logger.warning("⚠️ Strategy Manager не ініціалізовано")
            return
        
        try:
            self.strategy_manager.record_trade(symbol, pnl, strategy_name)
            logger.debug(f"📝 Записано угоду {symbol}: PnL=${pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Помилка запису угоди: {e}")
    
    def update_portfolio_value(self, new_value: float):
        """
        Оновлення вартості портфеля
        
        Args:
            new_value: Нова вартість портфеля
        """
        if not self.initialized or not self.strategy_manager:
            return
        
        try:
            self.portfolio_value = new_value
            self.strategy_manager.update_portfolio_value(new_value)
            logger.debug(f"💰 Портфель оновлено: ${new_value:.2f}")
        except Exception as e:
            logger.error(f"❌ Помилка оновлення портфеля: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Отримання статистики продуктивності
        
        Returns:
            Dict з метриками продуктивності
        """
        if not self.initialized or not self.strategy_manager:
            return {
                'initialized': False,
                'portfolio_value': self.portfolio_value,
                'total_trades': 0,
                'win_rate': 0.0
            }
        
        try:
            return self.strategy_manager.get_performance_summary()
        except Exception as e:
            logger.error(f"❌ Помилка отримання статистики: {e}")
            return {'error': str(e)}
    
    def reset_daily_stats(self):
        """Скидання денної статистики"""
        if not self.initialized or not self.strategy_manager:
            return
        
        try:
            self.strategy_manager.reset_daily_stats()
            logger.info("🔄 Денна статистика стратегій скинута")
        except Exception as e:
            logger.error(f"❌ Помилка скидання статистики: {e}")
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """
        Отримання всіх активних позицій
        
        Returns:
            Dict з активними позиціями по всіх стратегіях
        """
        if not self.initialized or not self.strategy_manager:
            return {}
        
        try:
            all_positions = {}
            for strategy_name, strategy in self.strategy_manager.strategies.items():
                for symbol, position in strategy.positions.items():
                    all_positions[f"{strategy_name}_{symbol}"] = {
                        'strategy': strategy_name,
                        'symbol': symbol,
                        'position': position
                    }
            return all_positions
        except Exception as e:
            logger.error(f"❌ Помилка отримання позицій: {e}")
            return {}
    
    def shutdown(self):
        """Коректне завершення роботи"""
        if self.strategy_manager:
            logger.info("🔄 Завершення роботи Strategy Integration...")
            # Тут можна додати логіку закриття позицій, якщо потрібно
        
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
