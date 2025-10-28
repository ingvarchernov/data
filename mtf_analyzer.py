#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Analyzer
Аналізує декілька таймфреймів для підвищення точності сигналів
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Аналізує декілька таймфреймів і комбінує сигнали
    
    Логіка:
    - 4h (primary): основний тренд і прогноз
    - 1h (confirmation): підтвердження тренду
    - 15m (timing): точка входу
    
    Комбінація:
    - Всі 3 таймфрейми в одному напрямку = висока впевненість
    - 2 з 3 = середня впевненість
    - Розбіжність = низька впевненість (не торгуємо)
    """
    
    def __init__(self):
        self.timeframes = {
            '4h': {'weight': 0.5, 'interval': '4h'},   # Основний тренд
            '1h': {'weight': 0.4, 'interval': '1h'},   # Підтвердження
            '15m': {'weight': 0.1, 'interval': '15m'}  # Точка входу (зменшено)
        }
    
    def analyze(
        self,
        predictions: Dict[str, Dict],  # {timeframe: {prediction, confidence, price}}
        require_alignment: bool = True
    ) -> Optional[Dict]:
        """
        Аналізує прогнози з різних таймфреймів
        
        Args:
            predictions: Dict з прогнозами для кожного таймфрейму
            require_alignment: Чи вимагати співпадіння напрямків
            
        Returns:
            Combined prediction або None якщо немає консенсусу
        """
        if not predictions:
            return None
        
        # Перевірка наявності всіх таймфреймів
        available_tf = set(predictions.keys())
        required_tf = set(self.timeframes.keys())
        
        if not required_tf.issubset(available_tf):
            missing = required_tf - available_tf
            logger.warning(f"⚠️ Відсутні таймфрейми: {missing}")
            return None
        
        # Збір напрямків і впевненостей
        directions = {}
        confidences = {}
        prices = {}
        
        for tf, pred in predictions.items():
            if tf not in self.timeframes:
                continue
            
            directions[tf] = pred['prediction']
            confidences[tf] = pred['confidence']
            prices[tf] = pred['current_price']
        
        # Підрахунок консенсусу
        up_weight = sum(
            self.timeframes[tf]['weight']
            for tf, direction in directions.items()
            if direction == 'UP'
        )
        
        down_weight = sum(
            self.timeframes[tf]['weight']
            for tf, direction in directions.items()
            if direction == 'DOWN'
        )
        
        # Визначення фінального напрямку
        if up_weight > down_weight:
            final_direction = 'UP'
            consensus_strength = up_weight
        elif down_weight > up_weight:
            final_direction = 'DOWN'
            consensus_strength = down_weight
        else:
            # Рівновага - не торгуємо
            logger.info("⚖️ MTF: рівновага між UP/DOWN")
            return None
        
        # Перевірка консенсусу (2/3 majority замість 100%)
        total_weight = sum(self.timeframes[tf]['weight'] for tf in directions.keys())
        consensus_threshold = total_weight * 0.6  # 60% для 2/3
        
        if consensus_strength < consensus_threshold:
            logger.info(
                f"⚠️ MTF: недостатній консенсус ({consensus_strength:.1f}/{total_weight:.1f}) - "
                f"4h:{directions.get('4h')}, "
                f"1h:{directions.get('1h')}, "
                f"15m:{directions.get('15m')}"
            )
            return None
        
        # Розрахунок зваженої впевненості
        weighted_confidence = sum(
            confidences[tf] * self.timeframes[tf]['weight']
            for tf in self.timeframes.keys()
            if tf in confidences
        )
        
        # Бонус за повне співпадіння
        if all(d == final_direction for d in directions.values()):
            alignment_bonus = 0.1  # +10% до впевненості
            weighted_confidence = min(1.0, weighted_confidence + alignment_bonus)
            logger.info(f"✨ MTF: всі таймфрейми співпадають - бонус +10%")
        
        # Використання ціни з 15m (найактуальніша)
        current_price = prices.get('15m', prices.get('1h', prices.get('4h')))
        
        result = {
            'prediction': final_direction,
            'confidence': weighted_confidence,
            'current_price': current_price,
            'consensus_strength': consensus_strength,
            'timeframe_signals': {
                tf: {
                    'direction': directions[tf],
                    'confidence': confidences[tf]
                }
                for tf in self.timeframes.keys()
            },
            'alignment': all(d == final_direction for d in directions.values())
        }
        
        # Логування
        self._log_analysis(result)
        
        return result
    
    def _log_analysis(self, result: Dict):
        """Логування результатів MTF аналізу"""
        direction_emoji = "📈" if result['prediction'] == 'UP' else "📉"
        
        log_msg = (
            f"\n{'='*60}\n"
            f"🔄 MULTI-TIMEFRAME ANALYSIS\n"
            f"{'='*60}\n"
            f"{direction_emoji} Напрямок: {result['prediction']}\n"
            f"💪 Впевненість: {result['confidence']:.2%}\n"
            f"🎯 Консенсус: {result['consensus_strength']:.1%}\n"
            f"✅ Співпадіння: {'ТАК' if result['alignment'] else 'НІ'}\n"
            f"\nСигнали по таймфреймам:\n"
        )
        
        for tf, signal in result['timeframe_signals'].items():
            emoji = "📈" if signal['direction'] == 'UP' else "📉"
            weight = self.timeframes[tf]['weight']
            log_msg += (
                f"  {tf:>3} ({weight:.0%}): {emoji} {signal['direction']:>4} "
                f"({signal['confidence']:.1%})\n"
            )
        
        log_msg += f"{'='*60}"
        
        logger.info(log_msg)
    
    def get_timeframe_data_requirements(self) -> Dict[str, int]:
        """
        Повертає вимоги до кількості свічок для кожного таймфрейму
        
        Returns:
            Dict з інтервалами та кількістю свічок
        """
        return {
            '4h': 500,   # ~83 дні
            '1h': 500,   # ~20 днів
            '15m': 500   # ~5 днів
        }


# Приклад використання
def example():
    """Приклад використання MTF аналізатора"""
    analyzer = MultiTimeframeAnalyzer()
    
    # Симуляція прогнозів з різних таймфреймів
    predictions = {
        '4h': {
            'prediction': 'UP',
            'confidence': 0.75,
            'current_price': 50000.0
        },
        '1h': {
            'prediction': 'UP',
            'confidence': 0.68,
            'current_price': 50010.0
        },
        '15m': {
            'prediction': 'UP',
            'confidence': 0.62,
            'current_price': 50015.0
        }
    }
    
    # Аналіз
    result = analyzer.analyze(predictions)
    
    if result:
        print(f"\nФінальний сигнал: {result['prediction']}")
        print(f"Впевненість: {result['confidence']:.2%}")
        print(f"Всі співпадають: {result['alignment']}")
    else:
        print("Немає чіткого сигналу")
    
    # Приклад розбіжності
    print("\n" + "="*60)
    print("Приклад з розбіжністю:")
    print("="*60)
    
    predictions_mixed = {
        '4h': {
            'prediction': 'UP',
            'confidence': 0.72,
            'current_price': 50000.0
        },
        '1h': {
            'prediction': 'DOWN',
            'confidence': 0.65,
            'current_price': 50010.0
        },
        '15m': {
            'prediction': 'UP',
            'confidence': 0.61,
            'current_price': 50015.0
        }
    }
    
    result = analyzer.analyze(predictions_mixed, require_alignment=True)
    
    if result:
        print(f"\nФінальний сигнал: {result['prediction']}")
    else:
        print("\n❌ Немає сигналу (розбіжність таймфреймів)")


if __name__ == '__main__':
    example()
