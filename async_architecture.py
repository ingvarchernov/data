# -*- coding: utf-8 -*-
"""
Виправлена асинхронна архітектура
"""
import asyncio
import logging
import gc
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
import tensorflow as tf

# ✅ ВИПРАВЛЕННЯ: Lazy import для уникнення циклічних залежностей
from cache_system import cache_manager, async_cached

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
            self.id = f"task_{int(datetime.now().timestamp() * 1000000)}"


class ResourceMonitor:
    """Монітор системних ресурсів"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.gpu_memory_threshold = 90.0  # %
    
    def get_system_status(self) -> Dict:
        """Отримання статусу системи"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
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
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return {'available': False}
            
            gpu_info = {
                'available': True,
                'count': len(gpus),
                'details': []
            }
            
            # ✅ ВИПРАВЛЕННЯ: Опціональний pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                
                for i in range(len(gpus)):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        name = pynvml.nvmlDeviceGetName(handle)
                        
                        gpu_info['details'].append({
                            'id': i,
                            'name': name.decode('utf-8') if isinstance(name, bytes) else name,
                            'memory_used_mb': memory_info.used / (1024**2),
                            'memory_total_mb': memory_info.total / (1024**2),
                            'memory_percent': (memory_info.used / memory_info.total) * 100,
                            'gpu_utilization': utilization.gpu,
                            'memory_utilization': utilization.memory
                        })
                    except Exception as e:
                        logger.warning(f"Не вдалося отримати інфо про GPU {i}: {e}")
                
                pynvml.nvmlShutdown()
                
            except ImportError:
                logger.debug("pynvml недоступний, використовується базова інформація про GPU")
                for i, gpu in enumerate(gpus):
                    gpu_info['details'].append({
                        'id': i,
                        'name': gpu.name,
                        'memory_total_mb': 'Unknown'
                    })
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Помилка отримання інформації про GPU: {e}")
            return {'available': False}
    
    def is_system_overloaded(self) -> bool:
        """Перевірка перевантаження системи"""
        status = self.get_system_status()
        
        if status.get('cpu_percent', 0) > self.cpu_threshold:
            logger.warning(f"⚠️ CPU перевантажено: {status['cpu_percent']:.1f}%")
            return True
        
        if status.get('memory_percent', 0) > self.memory_threshold:
            logger.warning(f"⚠️ Пам'ять перевантажена: {status['memory_percent']:.1f}%")
            return True
        
        gpu_info = status.get('gpu_info', {})
        if gpu_info.get('available'):
            for gpu in gpu_info.get('details', []):
                if gpu.get('memory_percent', 0) > self.gpu_memory_threshold:
                    logger.warning(f"⚠️ GPU {gpu['id']} перевантажено: {gpu['memory_percent']:.1f}%")
                    return True
        
        return False
    
    def can_allocate_resources(self, memory_mb: int, requires_gpu: bool = False) -> bool:
        """Перевірка можливості виділення ресурсів"""
        status = self.get_system_status()
        
        # Перевірка пам'яті (з запасом 20%)
        available_memory_mb = status.get('memory_available_gb', 0) * 1024 * 0.8
        if available_memory_mb < memory_mb:
            logger.debug(f"Недостатньо пам'яті: потрібно {memory_mb}MB, доступно {available_memory_mb:.0f}MB")
            return False
        
        # Перевірка GPU
        if requires_gpu:
            gpu_info = status.get('gpu_info', {})
            if not gpu_info.get('available'):
                logger.debug("GPU недоступний")
                return False
            
            # Перевірка вільної пам'яті GPU
            gpu_available = False
            for gpu in gpu_info.get('details', []):
                if gpu.get('memory_percent', 100) < self.gpu_memory_threshold:
                    gpu_available = True
                    break
            
            if not gpu_available:
                logger.debug("Всі GPU перевантажені")
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
                await asyncio.wait_for(queue.put(task), timeout=5.0)
                self._task_registry[task.id] = task
                logger.debug(f"📋 Задача {task.id} додана ({task.priority.name})")
                return True
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Таймаут додавання задачі {task.id}")
                return False
            except Exception as e:
                logger.error(f"❌ Помилка додавання задачі: {e}")
                return False
    
    async def get(self) -> Optional[Task]:
        """Отримання задачі з найвищим пріоритетом (non-blocking)"""
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
                logger.info(f"🛑 Задача {task_id} скасована")
                return True
            return False
    
    def get_queue_stats(self) -> Dict:
        """Статистика черг"""
        return {
            priority.name: {
                'pending': queue.qsize(),
                'maxsize': queue.maxsize
            }
            for priority, queue in self._queues.items()
        }
    
    def get_total_pending(self) -> int:
        """Загальна кількість задач в очікуванні"""
        return sum(q.qsize() for q in self._queues.values())


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
        
        # ✅ ВИПРАВЛЕННЯ: Правильна ініціалізація executors
        self.thread_executor = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="async_thread"
        )
        self.process_executor = ProcessPoolExecutor(max_workers=process_pool_size)
        
        # Статистика
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0
        }
        
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Запуск пулу воркерів"""
        if self.running:
            logger.warning("Worker pool вже запущено")
            return
        
        self.running = True
        logger.info(f"🚀 Запуск worker pool: {self.max_workers} воркерів")
        
        # Створюємо воркерів
        for i in range(self.max_workers):
            worker_id = f"worker-{i}-{id(asyncio.current_task()) % 1000:03d}"
            worker = asyncio.create_task(
                self._worker(worker_id),
                name=worker_id
            )
            self.workers.append(worker)
        
        # Запускаємо монітор ресурсів
        monitor_task = asyncio.create_task(
            self._resource_monitor_loop(),
            name="resource-monitor"
        )
        self.workers.append(monitor_task)
        
        logger.info(f"✅ {len(self.workers)} задач запущено")
    
    async def stop(self, timeout: float = 10.0):
        """Зупинка пулу воркерів"""
        if not self.running:
            return
        
        self.running = False
        logger.info("⏸️ Зупинка worker pool...")
        
        try:
            # ✅ ВИПРАВЛЕННЯ: Швидке завершення
            # 1. Скасовуємо всі воркери негайно
            for worker in self.workers:
                if not worker.done():
                    worker.cancel()
            
            # 2. Чекаємо завершення з коротким таймаутом
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Таймаут зупинки воркерів ({timeout}s)")
            
            # 3. Закриваємо executor pools з коротким таймаутом
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
            
            self.workers.clear()
            logger.info("✅ Worker pool зупинено")
            
        except Exception as e:
            logger.error(f"❌ Помилка зупинки worker pool: {e}")
    
    async def submit_task(self, 
                         func: Callable,
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[int] = None,
                         requires_gpu: bool = False,
                         memory_requirement: int = 1024,
                         **kwargs) -> str:
        """Додавання задачі до черги"""
        
        if not self.running:
            raise RuntimeError("Worker pool не запущено")
        
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
            logger.debug(f"📋 Задача {task.id} додана до черги")
            return task.id
        else:
            raise Exception("Не вдалося додати задачу до черги")
    
    async def get_task_result(self, task_id: str, timeout: Optional[int] = 60) -> Any:
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
            elapsed = (datetime.now() - start_time).total_seconds()
            if timeout and elapsed > timeout:
                raise TimeoutError(f"Таймаут очікування результату задачі {task_id} ({elapsed:.1f}s)")
            
            await asyncio.sleep(0.5)
    
    async def _worker(self, worker_name: str):
        """Воркер для виконання задач"""
        logger.info(f"👷 {worker_name} запущено")
        
        while self.running:
            try:
                # Отримуємо задачу
                task = await self.task_queue.get()
                if not task:
                    await asyncio.sleep(0.5)
                    continue
                
                # Перевіряємо ресурси
                retries = 0
                while not self.resource_monitor.can_allocate_resources(
                    task.memory_requirement, task.requires_gpu
                ):
                    if retries >= 3:
                        logger.warning(f"⚠️ Не вдалося виділити ресурси для {task.id}, повертаємо в чергу")
                        task.status = TaskStatus.PENDING
                        await self.task_queue.put(task)
                        break
                    
                    await asyncio.sleep(5)
                    retries += 1
                    continue
                
                if task.status != TaskStatus.RUNNING:
                    continue
                
                logger.debug(f"🔧 {worker_name} виконує {task.id}")
                
                # Виконуємо задачу
                start_time = datetime.now()
                try:
                    result = await self._execute_task(task)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    async with self._lock:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        task.completed_at = datetime.now()
                        
                        # Оновлюємо статистику
                        self.stats['tasks_completed'] += 1
                        self.stats['total_execution_time'] += execution_time
                        self.stats['avg_execution_time'] = (
                            self.stats['total_execution_time'] / self.stats['tasks_completed']
                        )
                    
                    logger.info(f"✅ {task.id} виконана за {execution_time:.2f}s")
                    
                except asyncio.CancelledError:
                    task.status = TaskStatus.CANCELLED
                    self.stats['tasks_cancelled'] += 1
                    raise
                    
                except Exception as e:
                    async with self._lock:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = datetime.now()
                        self.stats['tasks_failed'] += 1
                    
                    logger.error(f"❌ {task.id} провалена: {e}")
                    
                    # Повторні спроби
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.PENDING
                        await self.task_queue.put(task)
                        logger.info(f"🔄 Повторна спроба {task.retries}/{task.max_retries} для {task.id}")
                
                # Очищення пам'яті
                gc.collect()
                
            except asyncio.CancelledError:
                logger.info(f"👷 {worker_name} скасовано")
                break
            except Exception as e:
                logger.error(f"❌ Критична помилка в {worker_name}: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info(f"👷 {worker_name} завершено")
    
    async def _execute_task(self, task: Task) -> Any:
        """Виконання задачі"""
        func = task.func
        args = task.args
        kwargs = task.kwargs
        
        # Визначаємо тип виконання
        if asyncio.iscoroutinefunction(func):
            # Асинхронна функція
            if task.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=task.timeout
                )
            else:
                result = await func(*args, **kwargs)
        else:
            # Синхронна функція - виконуємо в executor
            loop = asyncio.get_running_loop()
            
            # Вибираємо executor
            if task.requires_gpu or getattr(func, '_use_thread_executor', False):
                executor = self.thread_executor
            else:
                executor = self.process_executor
            
            # ✅ ВИПРАВЛЕННЯ: Правильний виклик з executor
            if task.timeout:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: func(*args, **kwargs)),
                    timeout=task.timeout
                )
            else:
                result = await loop.run_in_executor(
                    executor, lambda: func(*args, **kwargs)
                )
        
        return result
    
    async def _resource_monitor_loop(self):
        """Цикл моніторингу ресурсів"""
        logger.info("📊 Монітор ресурсів запущено")
        
        while self.running:
            try:
                if self.resource_monitor.is_system_overloaded():
                    logger.warning("⚠️ Система перевантажена, пауза 10s")
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(30)
                    
                # Логуємо статистику кожні 5 хвилин
                if int(datetime.now().timestamp()) % 300 == 0:
                    stats = self.get_stats()
                    logger.info(
                        f"📊 Stats: {stats['tasks_completed']} completed, "
                        f"{stats['tasks_failed']} failed, "
                        f"{stats['pending_tasks']} pending"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Помилка моніторингу: {e}")
                await asyncio.sleep(30)
        
        logger.info("📊 Монітор ресурсів зупинено")
    
    def get_stats(self) -> Dict:
        """Статистика пулу воркерів"""
        stats = self.stats.copy()
        stats.update({
            'active_workers': sum(1 for w in self.workers if not w.done() and 'worker' in w.get_name()),
            'max_workers': self.max_workers,
            'pending_tasks': self.task_queue.get_total_pending(),
            'queue_stats': self.task_queue.get_queue_stats(),
            'system_status': self.resource_monitor.get_system_status()
        })
        return stats


class AsyncMLPipeline:
    """Асинхронний ML пайплайн"""
    
    def __init__(self):
        self.worker_pool = AsyncWorkerPool(
            max_workers=4,
            thread_pool_size=8,
            process_pool_size=2
        )
        self.initialized = False
    
    async def start(self):
        """Запуск пайплайну"""
        if self.initialized:
            logger.warning("ML Pipeline вже запущено")
            return
        
        await self.worker_pool.start()
        self.initialized = True
        logger.info("🚀 ML Pipeline запущено")
    
    async def stop(self):
        """Зупинка пайплайну"""
        if not self.initialized:
            return
        
        await self.worker_pool.stop(timeout=10.0)
        self.initialized = False
        logger.info("⏸️ ML Pipeline зупинено")
    
    # ✅ ВИПРАВЛЕННЯ: Видалено _train_model_task та _predict_prices_task
    # які використовували asyncio.run() всередині async контексту
    # Ці методи мають бути реалізовані в unified_intelligent_system


# Глобальний екземпляр
ml_pipeline = AsyncMLPipeline()


# ============================================================================
# ПУБЛІЧНИЙ API
# ============================================================================

async def init_async_system():
    """Ініціалізація асинхронної системи"""
    try:
        # ✅ ВИПРАВЛЕННЯ: Видалено uvloop.install() - це має бути в main.py
        await ml_pipeline.start()
        logger.info("✅ Асинхронна система ініціалізована")
    except Exception as e:
        logger.error(f"❌ Помилка ініціалізації async системи: {e}")
        raise


async def shutdown_async_system():
    """Завершення роботи асинхронної системи"""
    try:
        await ml_pipeline.stop()
        
        # ✅ ВИПРАВЛЕННЯ: Lazy import для уникнення циклічної залежності
        from optimized.database import DatabaseConnection
        db_manager = DatabaseConnection()
        await db_manager.close()
        
        logger.info("✅ Асинхронна система завершена")
    except Exception as e:
        logger.error(f"❌ Помилка shutdown async системи: {e}")