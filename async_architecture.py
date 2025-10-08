# -*- coding: utf-8 -*-
"""
Асинхронна архітектура з task queue, Worker pool та Resource Manager
Оптимізована для ML операцій і великих обсягів даних
"""
import asyncio
import logging
import os
import gc
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from pathlib import Path

import uvloop  # Швидша event loop реалізація
import redis.asyncio as aioredis
import numpy as np
import pandas as pd
import tensorflow as tf

# Спеціалізовані модулі
from cache_system import cache_manager, async_cached
from optimized_db import db_manager
from optimized_indicators import global_calculator
from optimized_model import OptimizedPricePredictionModel

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Пріоритети задач"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Статуси задач"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Задача для виконання"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    requires_gpu: bool = False
    memory_requirement: int = 1024  # MB
    
    def __post_init__(self):
        if not self.id:
            self.id = f"task_{int(datetime.now().timestamp() * 1000)}"

class ResourceMonitor:
    """Монітор системних ресурсів"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.gpu_memory_threshold = 90.0  # %
        
    def get_system_status(self) -> Dict:
        """Отримання статусу системи"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'gpu_info': self._get_gpu_info(),
                'timestamp': datetime.now()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Помилка моніторингу ресурсів: {e}")
            return {}
    
    def _get_gpu_info(self) -> Dict:
        """Інформація про GPU"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                return {'available': False}
            
            gpu_info = {'available': True, 'count': len(gpus)}
            
            # Спроба отримати інформацію про використання пам'яті
            try:
                import pynvml
                pynvml.nvmlInit()
                
                gpu_details = []
                for i in range(len(gpus)):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_details.append({
                        'id': i,
                        'memory_used_mb': memory_info.used / (1024**2),
                        'memory_total_mb': memory_info.total / (1024**2),
                        'memory_percent': (memory_info.used / memory_info.total) * 100,
                        'gpu_utilization': utilization.gpu,
                        'memory_utilization': utilization.memory
                    })
                
                gpu_info['details'] = gpu_details
                
            except ImportError:
                logger.warning("pynvml недоступний, детальна інформація про GPU недоступна")
                
            return gpu_info
            
        except Exception as e:
            logger.error(f"Помилка отримання інформації про GPU: {e}")
            return {'available': False}
    
    def is_system_overloaded(self) -> bool:
        """Перевірка перевантаження системи"""
        status = self.get_system_status()
        
        if status.get('cpu_percent', 0) > self.cpu_threshold:
            return True
        if status.get('memory_percent', 0) > self.memory_threshold:
            return True
        
        gpu_info = status.get('gpu_info', {})
        if gpu_info.get('available'):
            for gpu in gpu_info.get('details', []):
                if gpu.get('memory_percent', 0) > self.gpu_memory_threshold:
                    return True
        
        return False
    
    def can_allocate_resources(self, memory_mb: int, requires_gpu: bool = False) -> bool:
        """Перевірка можливості виділення ресурсів"""
        status = self.get_system_status()
        
        # Перевірка пам'яті
        available_memory_mb = status.get('memory_available_gb', 0) * 1024
        if available_memory_mb < memory_mb:
            return False
        
        # Перевірка GPU
        if requires_gpu:
            gpu_info = status.get('gpu_info', {})
            if not gpu_info.get('available'):
                return False
        
        return True

class TaskQueue:
    """Черга задач з пріоритетами"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._queues = {
            TaskPriority.CRITICAL: asyncio.Queue(maxsize=maxsize//4),
            TaskPriority.HIGH: asyncio.Queue(maxsize=maxsize//4),
            TaskPriority.NORMAL: asyncio.Queue(maxsize=maxsize//2),
            TaskPriority.LOW: asyncio.Queue(maxsize=maxsize//4)
        }
        self._task_registry: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
    
    async def put(self, task: Task) -> bool:
        """Додавання задачі в чергу"""
        async with self._lock:
            try:
                queue = self._queues[task.priority]
                await queue.put(task)
                self._task_registry[task.id] = task
                logger.debug(f"Задача {task.id} додана в чергу з пріоритетом {task.priority.name}")
                return True
            except asyncio.QueueFull:
                logger.error(f"Черга переповнена для пріоритету {task.priority.name}")
                return False
    
    async def get(self) -> Optional[Task]:
        """Отримання задачі з найвищим пріоритетом"""
        # Перевіряємо черги за пріоритетом
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self._queues[priority]
            try:
                task = queue.get_nowait()
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                return task
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Отримання статусу задачі"""
        task = self._task_registry.get(task_id)
        return task.status if task else None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Скасування задачі"""
        async with self._lock:
            task = self._task_registry.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
            return False
    
    def get_queue_stats(self) -> Dict:
        """Статистика черг"""
        return {
            priority.name: queue.qsize() 
            for priority, queue in self._queues.items()
        }

class AsyncWorkerPool:
    """Пул асинхронних воркерів"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 thread_pool_size: int = 8,
                 process_pool_size: int = 4):
        
        self.max_workers = max_workers
        self.workers: List[asyncio.Task] = []
        self.task_queue = TaskQueue()
        self.resource_monitor = ResourceMonitor()
        self.running = False
        
        # Executor pools для CPU-intensive задач
        self.thread_executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.process_executor = ProcessPoolExecutor(max_workers=process_pool_size)
        
        # Статистика
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0
        }
    
    async def start(self):
        """Запуск пулу воркерів"""
        if self.running:
            return
        
        self.running = True
        logger.info(f"🚀 Запуск пулу з {self.max_workers} воркерів")
        
        # Створюємо воркерів
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Запускаємо монітор ресурсів
        monitor_task = asyncio.create_task(self._resource_monitor_loop())
        self.workers.append(monitor_task)
    
    async def stop(self):
        """Зупинка пулу воркерів"""
        if not self.running:
            return
        
        self.running = False
        logger.info("⏸️ Зупинка пулу воркерів")
        
        # Скасовуємо всі задачі воркерів
        for worker in self.workers:
            worker.cancel()
        
        # Очікуємо завершення
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Закриваємо executor pools
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        self.workers.clear()
        logger.info("✅ Пул воркерів зупинено")
    
    async def submit_task(self, 
                         func: Callable,
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[int] = None,
                         requires_gpu: bool = False,
                         memory_requirement: int = 1024,
                         **kwargs) -> str:
        """Додавання задачі до черги"""
        
        task = Task(
            id=f"task_{int(datetime.now().timestamp() * 1000000)}",
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            requires_gpu=requires_gpu,
            memory_requirement=memory_requirement
        )
        
        success = await self.task_queue.put(task)
        if success:
            logger.info(f"📋 Задача {task.id} додана до черги")
            return task.id
        else:
            raise Exception("Не вдалося додати задачу до черги")
    
    async def get_task_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """Очікування результату задачі"""
        start_time = datetime.now()
        
        while True:
            task = self.task_queue._task_registry.get(task_id)
            if not task:
                raise ValueError(f"Задача {task_id} не знайдена")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise Exception(f"Задача провалена: {task.error}")
            elif task.status == TaskStatus.CANCELLED:
                raise Exception("Задача скасована")
            
            # Перевірка таймауту
            if timeout and (datetime.now() - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Таймаут очікування результату задачі {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def _worker(self, worker_name: str):
        """Воркер для виконання задач"""
        logger.info(f"👷 Воркер {worker_name} запущено")
        
        while self.running:
            try:
                # Отримуємо задачу
                task = await self.task_queue.get()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Перевіряємо ресурси
                if not self.resource_monitor.can_allocate_resources(
                    task.memory_requirement, task.requires_gpu):
                    # Повертаємо задачу в чергу
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                logger.info(f"🔧 Воркер {worker_name} виконує задачу {task.id}")
                
                # Виконуємо задачу
                start_time = datetime.now()
                try:
                    result = await self._execute_task(task)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now()
                    
                    # Оновлюємо статистику
                    self.stats['tasks_completed'] += 1
                    self.stats['total_execution_time'] += execution_time
                    self.stats['avg_execution_time'] = (
                        self.stats['total_execution_time'] / self.stats['tasks_completed']
                    )
                    
                    logger.info(f"✅ Задача {task.id} виконана за {execution_time:.2f}s")
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    
                    self.stats['tasks_failed'] += 1
                    logger.error(f"❌ Задача {task.id} провалена: {e}")
                    
                    # Повторні спроби
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.PENDING
                        await self.task_queue.put(task)
                        logger.info(f"🔄 Повторна спроба {task.retries}/{task.max_retries} для задачі {task.id}")
                
                # Очищення пам'яті
                gc.collect()
                
            except asyncio.CancelledError:
                logger.info(f"👷 Воркер {worker_name} скасовано")
                break
            except Exception as e:
                logger.error(f"❌ Помилка в воркері {worker_name}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> Any:
        """Виконання задачі"""
        func = task.func
        args = task.args
        kwargs = task.kwargs
        
        # Визначаємо тип виконання
        if asyncio.iscoroutinefunction(func):
            # Асинхронна функція
            if task.timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=task.timeout)
            else:
                result = await func(*args, **kwargs)
        else:
            # Синхронна функція - виконуємо в executor
            loop = asyncio.get_event_loop()
            
            # Вибираємо executor залежно від типу задачі
            if task.requires_gpu or hasattr(func, '_use_thread_executor'):
                executor = self.thread_executor
            else:
                executor = self.process_executor
            
            if task.timeout:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, func, *args, **kwargs),
                    timeout=task.timeout
                )
            else:
                result = await loop.run_in_executor(executor, func, *args, **kwargs)
        
        return result
    
    async def _resource_monitor_loop(self):
        """Цикл моніторингу ресурсів"""
        while self.running:
            try:
                if self.resource_monitor.is_system_overloaded():
                    logger.warning("⚠️ Система перевантажена, зменшуємо навантаження")
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Помилка моніторингу ресурсів: {e}")
                await asyncio.sleep(30)
    
    def get_stats(self) -> Dict:
        """Статистика пулу воркерів"""
        stats = self.stats.copy()
        stats.update({
            'active_workers': len([w for w in self.workers if not w.done()]),
            'queue_stats': self.task_queue.get_queue_stats(),
            'system_status': self.resource_monitor.get_system_status()
        })
        return stats

class AsyncMLPipeline:
    """Асинхронний ML пайплайн"""
    
    def __init__(self):
        self.worker_pool = AsyncWorkerPool(max_workers=6, thread_pool_size=12)
        self.pipeline_cache = {}
        
    async def start(self):
        """Запуск пайплайну"""
        await self.worker_pool.start()
        logger.info("🚀 ML пайплайн запущено")
    
    async def stop(self):
        """Зупинка пайплайну"""
        await self.worker_pool.stop()
        logger.info("⏸️ ML пайплайн зупинено")
    
    @async_cached(ttl=1800)
    async def process_data_async(self, symbol: str, interval: str, days_back: int) -> pd.DataFrame:
        """Асинхронна обробка даних"""
        # Отримання даних з БД
        data = await db_manager.get_historical_data_optimized(
            await db_manager.get_or_create_symbol_id(symbol),
            await db_manager.get_or_create_interval_id(interval),
            days_back
        )
        
        # Розрахунок індикаторів
        indicators = await global_calculator.calculate_all_indicators_batch(data)
        
        # Об'єднання даних
        for name, indicator in indicators.items():
            data[name] = indicator
        
        return data.dropna()
    
    async def train_model_async(self, 
                              symbol: str, 
                              interval: str, 
                              model_type: str = "transformer_lstm") -> str:
        """Асинхронне тренування моделі"""
        
        task_id = await self.worker_pool.submit_task(
            self._train_model_task,
            symbol, interval, model_type,
            priority=TaskPriority.HIGH,
            timeout=3600,
            requires_gpu=True,
            memory_requirement=4096
        )
        
        return task_id
    
    async def predict_prices_async(self, 
                                 symbol: str, 
                                 interval: str, 
                                 steps: int = 5) -> str:
        """Асинхронне прогнозування цін"""
        
        task_id = await self.worker_pool.submit_task(
            self._predict_prices_task,
            symbol, interval, steps,
            priority=TaskPriority.NORMAL,
            timeout=600,
            requires_gpu=True,
            memory_requirement=2048
        )
        
        return task_id
    
    def _train_model_task(self, symbol: str, interval: str, model_type: str) -> Dict:
        """Задача тренування моделі"""
        try:
            # Завантаження даних
            data = asyncio.run(self.process_data_async(symbol, interval, 365))
            
            # Підготовка даних для тренування
            # ... (логіка підготовки)
            
            # Створення та тренування моделі
            model_builder = OptimizedPricePredictionModel(
                input_shape=(360, len(data.columns)-1),
                model_type=model_type
            )
            
            # ... (логіка тренування)
            
            return {
                'status': 'completed',
                'model_path': f'models/{symbol}_{interval}_{model_type}.keras',
                'metrics': {'loss': 0.01, 'mae': 0.005}
            }
            
        except Exception as e:
            logger.error(f"Помилка тренування моделі: {e}")
            raise
    
    def _predict_prices_task(self, symbol: str, interval: str, steps: int) -> Dict:
        """Задача прогнозування цін"""
        try:
            # Завантаження даних та моделі
            # ... (логіка прогнозування)
            
            predictions = [100.5, 101.2, 99.8, 102.1, 103.5]  # Заглушка
            
            return {
                'symbol': symbol,
                'interval': interval,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Помилка прогнозування: {e}")
            raise

# Глобальні екземпляри
ml_pipeline = AsyncMLPipeline()

# Зручні функції
async def init_async_system():
    """Ініціалізація асинхронної системи"""
    # Встановлюємо uvloop для кращої продуктивності
    if hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    await ml_pipeline.start()
    logger.info("🚀 Асинхронна система ініціалізована")

async def shutdown_async_system():
    """Завершення роботи асинхронної системи"""
    await ml_pipeline.stop()
    await db_manager.close()
    logger.info("✅ Асинхронна система завершена")

def run_async_function(coro):
    """Запуск асинхронної функції"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Якщо цикл вже запущено, створюємо задачу
            return asyncio.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # Створюємо новий цикл
        return asyncio.run(coro)