# -*- coding: utf-8 -*-
"""
Система моніторингу для крипто-прогнозування
Збирає метрики продуктивності, стану системи та логів
"""
import time
import psutil
import GPUtil
from datetime import datetime
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Метрики стану системи"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None

@dataclass
class ModelMetrics:
    """Метрики моделі"""
    symbol: str
    interval: str
    model_type: str
    timestamp: datetime
    training_time: float
    epochs: int
    final_loss: float
    final_val_loss: float
    final_mae: float
    final_val_mae: float
    directional_accuracy: float
    mape: float
    val_mape: float

@dataclass
class PredictionMetrics:
    """Метрики прогнозів"""
    symbol: str
    interval: str
    timestamp: datetime
    predicted_price: float
    actual_price: float
    prediction_error: float
    directional_correct: bool
    confidence_score: Optional[float] = None

class MonitoringSystem:
    """Основна система моніторингу"""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.system_metrics: List[SystemMetrics] = []
        self.model_metrics: List[ModelMetrics] = []
        self.prediction_metrics: List[PredictionMetrics] = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_monitoring = False

    async def start_monitoring(self, interval_seconds: int = 60):
        """Запуск фонового моніторингу"""
        self.is_monitoring = True
        logger.info(f"🚀 Запуск системи моніторингу (інтервал: {interval_seconds}с)")

        while self.is_monitoring:
            try:
                metrics = await self._collect_system_metrics()
                self.system_metrics.append(metrics)

                # Зберігаємо в БД якщо є з'єднання
                if self.db_manager:
                    await self._save_system_metrics_to_db(metrics)

                # Обмежуємо розмір списків (зберігаємо останні 1000 записів)
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-500:]

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"❌ Помилка моніторингу: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Зупинка моніторингу"""
        self.is_monitoring = False
        self.executor.shutdown(wait=False)
        logger.info("⏸️ Система моніторингу зупинена")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Збір системних метрик"""
        def _get_gpu_info():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    return {
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'utilization': gpu.load * 100
                    }
            except Exception as e:
                logger.debug(f"Не вдалося отримати GPU інформацію: {e}")
            return None

        # Збираємо метрики в окремому потоці щоб не блокувати event loop
        loop = asyncio.get_event_loop()
        gpu_info = await loop.run_in_executor(self.executor, _get_gpu_info)

        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            gpu_memory_used=gpu_info['memory_used'] if gpu_info else None,
            gpu_memory_total=gpu_info['memory_total'] if gpu_info else None,
            gpu_utilization=gpu_info['utilization'] if gpu_info else None
        )

        return metrics

    async def _save_system_metrics_to_db(self, metrics: SystemMetrics):
        """Збереження системних метрик в БД"""
        try:
            data = {
                'timestamp': metrics.timestamp,
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage': metrics.disk_usage,
                'gpu_memory_used': metrics.gpu_memory_used,
                'gpu_memory_total': metrics.gpu_memory_total,
                'gpu_utilization': metrics.gpu_utilization
            }

            await self.db_manager.batch_insert('system_metrics', [data])
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося зберегти системні метрики: {e}")

    def record_model_metrics(self, symbol: str, interval: str, model_type: str,
                           training_time: float, history):
        """Запис метрик моделі"""
        try:
            # Отримуємо фінальні метрики з історії тренування
            # History object має атрибут history, який є dict
            hist_dict = history.history if hasattr(history, 'history') else history
            
            epochs = len(hist_dict.get('loss', []))
            final_loss = hist_dict.get('loss', [-1])[-1]
            final_val_loss = hist_dict.get('val_loss', [-1])[-1]
            final_mae = hist_dict.get('mae', [-1])[-1]
            final_val_mae = hist_dict.get('val_mae', [-1])[-1]
            directional_accuracy = hist_dict.get('directional_accuracy', [0])[-1]
            mape = hist_dict.get('mape', [0])[-1]
            val_mape = hist_dict.get('val_mape', [0])[-1]

            metrics = ModelMetrics(
                symbol=symbol,
                interval=interval,
                model_type=model_type,
                timestamp=datetime.now(),
                training_time=training_time,
                epochs=epochs,
                final_loss=final_loss,
                final_val_loss=final_val_loss,
                final_mae=final_mae,
                final_val_mae=final_val_mae,
                directional_accuracy=directional_accuracy,
                mape=mape,
                val_mape=val_mape
            )

            self.model_metrics.append(metrics)

            # Зберігаємо в БД
            if self.db_manager:
                asyncio.create_task(self._save_model_metrics_to_db(metrics))

            logger.info(f"📊 Метрики моделі {symbol} ({model_type}): "
                       f"MAE={final_val_mae:.4f}, MAPE={val_mape:.2f}%, "
                       f"Напрямок={directional_accuracy:.1f}%")

        except Exception as e:
            logger.error(f"❌ Помилка запису метрик моделі: {e}")

    async def _save_model_metrics_to_db(self, metrics: ModelMetrics):
        """Збереження метрик моделі в БД"""
        try:
            data = {
                'symbol': metrics.symbol,
                'interval': metrics.interval,
                'model_type': metrics.model_type,
                'timestamp': metrics.timestamp,
                'training_time': metrics.training_time,
                'epochs': metrics.epochs,
                'final_loss': metrics.final_loss,
                'final_val_loss': metrics.final_val_loss,
                'final_mae': metrics.final_mae,
                'final_val_mae': metrics.final_val_mae,
                'directional_accuracy': metrics.directional_accuracy,
                'mape': metrics.mape,
                'val_mape': metrics.val_mape
            }

            await self.db_manager.batch_insert('model_metrics', [data])
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося зберегти метрики моделі: {e}")

    def record_prediction_metrics(self, symbol: str, interval: str,
                                predicted_price: float, actual_price: float,
                                confidence_score: Optional[float] = None):
        """Запис метрик прогнозів"""
        try:
            prediction_error = abs(predicted_price - actual_price) / actual_price * 100
            directional_correct = (predicted_price - actual_price) * (actual_price - actual_price) >= 0  # порівняння з попередньою ціною

            metrics = PredictionMetrics(
                symbol=symbol,
                interval=interval,
                timestamp=datetime.now(),
                predicted_price=predicted_price,
                actual_price=actual_price,
                prediction_error=prediction_error,
                directional_correct=directional_correct,
                confidence_score=confidence_score
            )

            self.prediction_metrics.append(metrics)

            # Зберігаємо в БД
            if self.db_manager:
                asyncio.create_task(self._save_prediction_metrics_to_db(metrics))

        except Exception as e:
            logger.error(f"❌ Помилка запису метрик прогнозу: {e}")

    async def _save_prediction_metrics_to_db(self, metrics: PredictionMetrics):
        """Збереження метрик прогнозів в БД"""
        try:
            data = {
                'symbol': metrics.symbol,
                'interval': metrics.interval,
                'timestamp': metrics.timestamp,
                'predicted_price': metrics.predicted_price,
                'actual_price': metrics.actual_price,
                'prediction_error': metrics.prediction_error,
                'directional_correct': metrics.directional_correct,
                'confidence_score': metrics.confidence_score
            }

            await self.db_manager.batch_insert('prediction_metrics', [data])
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося зберегти метрики прогнозу: {e}")

    def get_system_status(self) -> Dict:
        """Отримання поточного стану системи"""
        try:
            latest_metrics = self.system_metrics[-1] if self.system_metrics else None

            status = {
                'monitoring_active': self.is_monitoring,
                'total_system_metrics': len(self.system_metrics),
                'total_model_metrics': len(self.model_metrics),
                'total_prediction_metrics': len(self.prediction_metrics),
                'current_metrics': None
            }

            if latest_metrics:
                status['current_metrics'] = {
                    'timestamp': latest_metrics.timestamp.isoformat(),
                    'cpu_percent': latest_metrics.cpu_percent,
                    'memory_percent': latest_metrics.memory_percent,
                    'disk_usage': latest_metrics.disk_usage,
                    'gpu_memory_used': latest_metrics.gpu_memory_used,
                    'gpu_memory_total': latest_metrics.gpu_memory_total,
                    'gpu_utilization': latest_metrics.gpu_utilization
                }

            return status

        except Exception as e:
            logger.error(f"❌ Помилка отримання статусу системи: {e}")
            return {'error': str(e)}

    def get_performance_summary(self) -> Dict:
        """Підсумок продуктивності"""
        try:
            summary = {
                'model_performance': {},
                'prediction_accuracy': {},
                'system_health': {}
            }

            # Аналіз метрик моделей
            if self.model_metrics:
                model_types = {}
                for m in self.model_metrics:
                    if m.model_type not in model_types:
                        model_types[m.model_type] = []
                    model_types[m.model_type].append(m)

                for model_type, metrics in model_types.items():
                    summary['model_performance'][model_type] = {
                        'avg_mae': sum(m.final_val_mae for m in metrics) / len(metrics),
                        'avg_mape': sum(m.val_mape for m in metrics) / len(metrics),
                        'avg_directional_accuracy': sum(m.directional_accuracy for m in metrics) / len(metrics),
                        'avg_training_time': sum(m.training_time for m in metrics) / len(metrics),
                        'total_models': len(metrics)
                    }

            # Аналіз точності прогнозів
            if self.prediction_metrics:
                total_predictions = len(self.prediction_metrics)
                correct_directional = sum(1 for m in self.prediction_metrics if m.directional_correct)
                avg_error = sum(m.prediction_error for m in self.prediction_metrics) / total_predictions

                summary['prediction_accuracy'] = {
                    'total_predictions': total_predictions,
                    'directional_accuracy': correct_directional / total_predictions * 100,
                    'avg_prediction_error_percent': avg_error
                }

            # Стан системи
            if self.system_metrics:
                recent_metrics = self.system_metrics[-10:]  # останні 10 записів
                summary['system_health'] = {
                    'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                    'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                    'avg_disk_usage': sum(m.disk_usage for m in recent_metrics) / len(recent_metrics),
                    'gpu_available': any(m.gpu_memory_total for m in recent_metrics if m.gpu_memory_total)
                }

            return summary

        except Exception as e:
            logger.error(f"❌ Помилка створення підсумку продуктивності: {e}")
            return {'error': str(e)}

# Глобальний екземпляр системи моніторингу
monitoring_system = MonitoringSystem()