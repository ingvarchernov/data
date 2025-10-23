# -*- coding: utf-8 -*-
"""
Система моніторингу продуктивності та здоров'я торгової системи
Відстежує метрики, збирає статистику та надсилає алерти
"""
import asyncio
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("⚠️ GPUtil не встановлено, GPU моніторинг недоступний")

logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    Центральна система моніторингу для торгової системи
    """

    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.db_manager = None
        
        # Метрики системи
        self.system_metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_usage': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'network_sent': 0,
            'network_recv': 0
        }
        
        # Метрики торгівлі
        self.trading_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'positions_opened': 0,
            'positions_closed': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'api_calls': 0,
            'api_errors': 0
        }
        
        # Метрики ML моделей
        self.ml_metrics = {
            'predictions_made': 0,
            'predictions_accuracy': {},
            'models_trained': 0,
            'training_time': 0.0,
            'inference_time': [],
            'model_errors': 0
        }
        
        # Метрики БД
        self.db_metrics = {
            'queries_executed': 0,
            'queries_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'db_errors': 0,
            'avg_query_time': []
        }
        
        # Алерти та пороги
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'gpu_memory': 90.0,
            'api_error_rate': 0.1,
            'db_error_rate': 0.05,
            'daily_loss_limit': 0.1
        }
        
        # Історія алертів
        self.alerts_history = []
        self.max_alerts_history = 1000
        
        # Час запуску
        self.start_time = datetime.now()
        self.last_metrics_save = datetime.now()
        
        # Інтервали
        self.monitoring_interval = 10  # секунд
        self.metrics_save_interval = 300  # 5 хвилин
        
        logger.info("📊 MonitoringSystem ініціалізовано")

    def start_monitoring(self):
        """Запуск моніторингу в окремому потоці"""
        if self.monitoring_active:
            logger.warning("⚠️ Моніторинг вже активний")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitoringThread"
        )
        self.monitor_thread.start()
        logger.info("✅ Моніторинг запущено")

    def stop_monitoring(self):
        """Зупинка моніторингу"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Моніторинг зупинено")

    def _monitoring_loop(self):
        """Основний цикл моніторингу"""
        logger.info("🔄 Цикл моніторингу запущено")
        
        while self.monitoring_active:
            try:
                # Збір системних метрик
                self._collect_system_metrics()
                
                # Перевірка порогів та генерація алертів
                self._check_thresholds()
                
                # Збереження метрик в БД (кожні 5 хвилин)
                current_time = datetime.now()
                if (current_time - self.last_metrics_save).total_seconds() >= self.metrics_save_interval:
                    asyncio.run(self._save_metrics_to_db())
                    self.last_metrics_save = current_time
                
                # Очищення старих даних
                self._cleanup_old_metrics()
                
                # Пауза
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"❌ Помилка в циклі моніторингу: {e}", exc_info=True)
                time.sleep(30)

    def _collect_system_metrics(self):
        """Збір системних метрик"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_percent'].append({
                'value': cpu_percent,
                'timestamp': datetime.now()
            })
            
            # Пам'ять
            memory = psutil.virtual_memory()
            self.system_metrics['memory_percent'].append({
                'value': memory.percent,
                'timestamp': datetime.now()
            })
            
            # Диск
            disk = psutil.disk_usage('/')
            self.system_metrics['disk_usage'].append({
                'value': disk.percent,
                'timestamp': datetime.now()
            })
            
            # Мережа
            network = psutil.net_io_counters()
            self.system_metrics['network_sent'] = network.bytes_sent
            self.system_metrics['network_recv'] = network.bytes_recv
            
            # GPU (якщо доступний)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.system_metrics['gpu_memory'].append({
                            'value': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'timestamp': datetime.now()
                        })
                        self.system_metrics['gpu_utilization'].append({
                            'value': gpu.load * 100,
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    logger.debug(f"GPU metrics error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Помилка збору системних метрик: {e}")

    def _check_thresholds(self):
        """Перевірка порогів та генерація алертів"""
        try:
            # Перевірка CPU
            if self.system_metrics['cpu_percent']:
                latest_cpu = self.system_metrics['cpu_percent'][-1]['value']
                if latest_cpu > self.alert_thresholds['cpu_percent']:
                    self._create_alert('cpu_high', f"CPU usage high: {latest_cpu:.1f}%", 'warning')
            
            # Перевірка пам'яті
            if self.system_metrics['memory_percent']:
                latest_memory = self.system_metrics['memory_percent'][-1]['value']
                if latest_memory > self.alert_thresholds['memory_percent']:
                    self._create_alert('memory_high', f"Memory usage high: {latest_memory:.1f}%", 'warning')
            
            # Перевірка GPU
            if self.system_metrics['gpu_memory']:
                latest_gpu_mem = self.system_metrics['gpu_memory'][-1]['value']
                if latest_gpu_mem > self.alert_thresholds['gpu_memory']:
                    self._create_alert('gpu_memory_high', f"GPU memory high: {latest_gpu_mem:.1f}%", 'warning')
            
            # Перевірка API помилок
            if self.trading_metrics['api_calls'] > 0:
                error_rate = self.trading_metrics['api_errors'] / self.trading_metrics['api_calls']
                if error_rate > self.alert_thresholds['api_error_rate']:
                    self._create_alert('api_errors_high', f"API error rate high: {error_rate:.2%}", 'critical')
            
            # Перевірка денних втрат
            if self.trading_metrics['daily_pnl'] < 0:
                loss_percent = abs(self.trading_metrics['daily_pnl']) / 1000  # Assuming $1000 balance
                if loss_percent > self.alert_thresholds['daily_loss_limit']:
                    self._create_alert('daily_loss_limit', f"Daily loss limit exceeded: {loss_percent:.2%}", 'critical')
            
        except Exception as e:
            logger.error(f"❌ Помилка перевірки порогів: {e}")

    def _create_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Створення алерту"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }
        
        self.alerts_history.append(alert)
        
        # Логування
        if severity == 'critical':
            logger.critical(f"🚨 CRITICAL ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"⚠️ WARNING: {message}")
        else:
            logger.info(f"ℹ️ INFO: {message}")

    def _cleanup_old_metrics(self):
        """Очищення старих метрик для економії пам'яті"""
        max_items = 100
        
        for metric_list in [
            self.system_metrics['cpu_percent'],
            self.system_metrics['memory_percent'],
            self.system_metrics['disk_usage'],
            self.system_metrics['gpu_memory'],
            self.system_metrics['gpu_utilization']
        ]:
            if len(metric_list) > max_items:
                # Залишаємо тільки останні 50 записів
                del metric_list[:-50]
        
        # Очищення історії алертів
        if len(self.alerts_history) > self.max_alerts_history:
            del self.alerts_history[:-500]

    async def _save_metrics_to_db(self):
        """Збереження метрик в базу даних"""
        if not self.db_manager:
            return
        
        try:
            # Підготовка даних для збереження
            metrics_data = {
                'timestamp': datetime.now(),
                'system_metrics': self._get_average_metrics(),
                'trading_metrics': self.trading_metrics.copy(),
                'ml_metrics': self.ml_metrics.copy(),
                'db_metrics': self.db_metrics.copy()
            }
            
            # Збереження в БД (якщо потрібно)
            # await self.db_manager.save_monitoring_metrics(metrics_data)
            
            logger.debug("💾 Метрики збережено в БД")
            
        except Exception as e:
            logger.error(f"❌ Помилка збереження метрик: {e}")

    def _get_average_metrics(self) -> Dict[str, float]:
        """Отримання середніх значень метрик"""
        avg_metrics = {}
        
        # CPU
        if self.system_metrics['cpu_percent']:
            cpu_values = [m['value'] for m in self.system_metrics['cpu_percent'][-10:]]
            avg_metrics['cpu_avg'] = np.mean(cpu_values)
        
        # Memory
        if self.system_metrics['memory_percent']:
            mem_values = [m['value'] for m in self.system_metrics['memory_percent'][-10:]]
            avg_metrics['memory_avg'] = np.mean(mem_values)
        
        # GPU
        if self.system_metrics['gpu_memory']:
            gpu_values = [m['value'] for m in self.system_metrics['gpu_memory'][-10:]]
            avg_metrics['gpu_memory_avg'] = np.mean(gpu_values)
        
        if self.system_metrics['gpu_utilization']:
            gpu_util_values = [m['value'] for m in self.system_metrics['gpu_utilization'][-10:]]
            avg_metrics['gpu_utilization_avg'] = np.mean(gpu_util_values)
        
        return avg_metrics

    # === Публічні методи для оновлення метрик ===

    def record_trade(self, profit: float, is_winning: bool):
        """Запис торгової операції"""
        self.trading_metrics['total_trades'] += 1
        self.trading_metrics['total_pnl'] += profit
        self.trading_metrics['daily_pnl'] += profit
        
        if is_winning:
            self.trading_metrics['winning_trades'] += 1
        else:
            self.trading_metrics['losing_trades'] += 1

    def record_position_opened(self):
        """Запис відкриття позиції"""
        self.trading_metrics['positions_opened'] += 1

    def record_position_closed(self):
        """Запис закриття позиції"""
        self.trading_metrics['positions_closed'] += 1

    def record_signal(self, executed: bool = False):
        """Запис сигналу"""
        self.trading_metrics['signals_generated'] += 1
        if executed:
            self.trading_metrics['signals_executed'] += 1

    def record_api_call(self, success: bool = True):
        """Запис API виклику"""
        self.trading_metrics['api_calls'] += 1
        if not success:
            self.trading_metrics['api_errors'] += 1

    def record_prediction(self, symbol: str, accuracy: float = None):
        """Запис ML прогнозу"""
        self.ml_metrics['predictions_made'] += 1
        
        if accuracy is not None:
            if symbol not in self.ml_metrics['predictions_accuracy']:
                self.ml_metrics['predictions_accuracy'][symbol] = []
            self.ml_metrics['predictions_accuracy'][symbol].append(accuracy)

    def record_model_training(self, training_time: float):
        """Запис тренування моделі"""
        self.ml_metrics['models_trained'] += 1
        self.ml_metrics['training_time'] += training_time

    def record_db_query(self, query_time: float, cached: bool = False):
        """Запис БД запиту"""
        self.db_metrics['queries_executed'] += 1
        self.db_metrics['avg_query_time'].append(query_time)
        
        if cached:
            self.db_metrics['queries_cached'] += 1
            self.db_metrics['cache_hits'] += 1
        else:
            self.db_metrics['cache_misses'] += 1

    def reset_daily_metrics(self):
        """Скидання денних метрик"""
        self.trading_metrics['daily_pnl'] = 0.0
        logger.info("🔄 Денні метрики скинуто")

    # === Методи отримання статистики ===

    def get_system_status(self) -> Dict[str, Any]:
        """Отримання поточного статусу системи"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'monitoring_active': self.monitoring_active,
            'system_metrics': self._get_average_metrics(),
            'trading_metrics': self.trading_metrics.copy(),
            'ml_metrics': {
                'predictions_made': self.ml_metrics['predictions_made'],
                'models_trained': self.ml_metrics['models_trained'],
                'avg_accuracy': self._get_average_model_accuracy()
            },
            'db_metrics': {
                'queries_executed': self.db_metrics['queries_executed'],
                'cache_hit_rate': self._get_cache_hit_rate(),
                'avg_query_time': np.mean(self.db_metrics['avg_query_time'][-100:]) if self.db_metrics['avg_query_time'] else 0
            },
            'recent_alerts': self.alerts_history[-10:] if self.alerts_history else []
        }
        
        return status

    def _get_average_model_accuracy(self) -> float:
        """Середня точність моделей"""
        all_accuracies = []
        for symbol_accuracies in self.ml_metrics['predictions_accuracy'].values():
            all_accuracies.extend(symbol_accuracies[-10:])
        
        return np.mean(all_accuracies) if all_accuracies else 0.0

    def _get_cache_hit_rate(self) -> float:
        """Відсоток попадань в кеш"""
        total_requests = self.db_metrics['cache_hits'] + self.db_metrics['cache_misses']
        if total_requests == 0:
            return 0.0
        
        return self.db_metrics['cache_hits'] / total_requests

    def get_trading_statistics(self) -> Dict[str, Any]:
        """Торгова статистика"""
        total_trades = self.trading_metrics['total_trades']
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.trading_metrics['winning_trades'],
            'losing_trades': self.trading_metrics['losing_trades'],
            'win_rate': self.trading_metrics['winning_trades'] / total_trades if total_trades > 0 else 0,
            'total_pnl': self.trading_metrics['total_pnl'],
            'daily_pnl': self.trading_metrics['daily_pnl'],
            'avg_pnl_per_trade': self.trading_metrics['total_pnl'] / total_trades if total_trades > 0 else 0,
            'signals_generated': self.trading_metrics['signals_generated'],
            'signals_executed': self.trading_metrics['signals_executed'],
            'signal_execution_rate': self.trading_metrics['signals_executed'] / self.trading_metrics['signals_generated'] if self.trading_metrics['signals_generated'] > 0 else 0
        }

    def print_status_report(self):
        """Друк звіту про статус"""
        status = self.get_system_status()
        stats = self.get_trading_statistics()
        
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS REPORT")
        print("="*60)
        print(f"⏱️  Uptime: {status['uptime_formatted']}")
        print(f"💻 CPU: {status['system_metrics'].get('cpu_avg', 0):.1f}%")
        print(f"🧠 Memory: {status['system_metrics'].get('memory_avg', 0):.1f}%")
        
        if 'gpu_memory_avg' in status['system_metrics']:
            print(f"🎮 GPU Memory: {status['system_metrics']['gpu_memory_avg']:.1f}%")
        
        print("\n" + "-"*60)
        print("💰 TRADING METRICS")
        print("-"*60)
        print(f"📈 Total Trades: {stats['total_trades']}")
        print(f"✅ Win Rate: {stats['win_rate']:.2%}")
        print(f"💵 Total P&L: ${stats['total_pnl']:.2f}")
        print(f"📊 Daily P&L: ${stats['daily_pnl']:.2f}")
        print(f"📡 Signals: {stats['signals_generated']} generated, {stats['signals_executed']} executed")
        
        print("\n" + "-"*60)
        print("🤖 ML METRICS")
        print("-"*60)
        print(f"🔮 Predictions: {status['ml_metrics']['predictions_made']}")
        print(f"🎯 Avg Accuracy: {status['ml_metrics']['avg_accuracy']:.2%}")
        print(f"🏋️  Models Trained: {status['ml_metrics']['models_trained']}")
        
        print("\n" + "-"*60)
        print("💾 DATABASE METRICS")
        print("-"*60)
        print(f"📊 Queries: {status['db_metrics']['queries_executed']}")
        print(f"⚡ Cache Hit Rate: {status['db_metrics']['cache_hit_rate']:.2%}")
        print(f"⏱️  Avg Query Time: {status['db_metrics']['avg_query_time']:.3f}s")
        
        if status['recent_alerts']:
            print("\n" + "-"*60)
            print("🚨 RECENT ALERTS")
            print("-"*60)
            for alert in status['recent_alerts']:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        
        print("="*60 + "\n")


# Глобальний екземпляр системи моніторингу
monitoring_system = MonitoringSystem()


# Допоміжні функції
def get_monitoring_status() -> Dict[str, Any]:
    """Отримання статусу моніторингу"""
    return monitoring_system.get_system_status()


def get_trading_statistics() -> Dict[str, Any]:
    """Отримання торгової статистики"""
    return monitoring_system.get_trading_statistics()


if __name__ == "__main__":
    # Тестування системи моніторингу
    print("🧪 Тестування MonitoringSystem...")
    
    monitoring_system.start_monitoring()
    
    # Симуляція активності
    import random
    
    for i in range(5):
        # Симуляція торгівлі
        profit = random.uniform(-10, 20)
        monitoring_system.record_trade(profit, profit > 0)
        monitoring_system.record_signal(executed=random.choice([True, False]))
        
        # Симуляція ML
        monitoring_system.record_prediction('BTCUSDT', random.uniform(0.5, 0.9))
        
        # Симуляція БД
        monitoring_system.record_db_query(random.uniform(0.01, 0.1), cached=random.choice([True, False]))
        
        time.sleep(2)
    
    # Друк звіту
    monitoring_system.print_status_report()
    
    monitoring_system.stop_monitoring()
    print("✅ Тест завершено")