# -*- coding: utf-8 -*-
"""
GPU конфігурація та утиліти
"""
from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


def _format_cuda_version(raw_version: int) -> str:
    """Перетворює версію CUDA з NVML (наприклад, 12020) у формат X.Y"""
    major = raw_version // 1000
    minor = (raw_version % 1000) // 10
    return f"{major}.{minor}"


def _reset_to_cpu_policy(enable_logging: bool = True) -> None:
    """Скидає mixed precision до float32 для стабільної роботи на CPU."""
    try:
        from tensorflow import keras

        current_policy = keras.mixed_precision.global_policy().name
        if current_policy != 'float32':
            keras.mixed_precision.set_global_policy('float32')
            if enable_logging:
                logger.info("ℹ️ Mixed precision вимкнено для CPU режиму")
    except Exception as err:  # pragma: no cover - політики можуть бути недоступними у тестах
        logger.debug("Не вдалося скинути mixed precision: %s", err)


def _log_gpu_environment_diagnostics(reason: str, error: Optional[Exception] = None) -> None:
    """Виводить додаткову діагностику, якщо TensorFlow не може використовувати GPU."""
    logger.warning("⚠️ TensorFlow не може використати GPU (%s)", reason)
    if error is not None:
        logger.warning("   ↳ %s", error)

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')

            device_count = pynvml.nvmlDeviceGetCount()
            logger.warning("🧩 NVML: виявлено %d GPU, драйвер %s", device_count, driver_version)

            cuda_driver_version = None
            if hasattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2"):
                cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            elif hasattr(pynvml, "nvmlSystemGetCudaDriverVersion"):
                cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()

            if cuda_driver_version:
                logger.warning(
                    "🧩 NVML: версія CUDA драйвера %s",
                    _format_cuda_version(cuda_driver_version),
                )

            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                logger.info(
                    "   • GPU %d: %s, пам'ять %.0f / %.0f MB",
                    index,
                    name,
                    memory.used / (1024 ** 2),
                    memory.total / (1024 ** 2),
                )
        finally:  # pragma: no cover - nvmlShutdown може не бути викликано у мокованому середовищі
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        logger.warning("pynvml недоступний, пропускаємо NVML діагностику")
    except Exception as nvml_error:
        logger.warning("Не вдалося зібрати NVML діагностику: %s", nvml_error)

    try:
        build_info = tf.sysconfig.get_build_info()
        logger.warning(
            "ℹ️ TensorFlow зібрано з CUDA %s / cuDNN %s",
            build_info.get('cuda_version', 'невідомо'),
            build_info.get('cudnn_version', 'невідомо'),
        )
    except Exception as build_error:
        logger.debug("Не вдалося зчитати TensorFlow build info: %s", build_error)

    logger.warning(
        "💡 Переконайтесь, що версія драйвера NVIDIA сумісна з TensorFlow та CUDA. "
        "Оновіть драйвер або встановіть відповідний пакет CUDA, потім перезапустіть систему."
    )


def configure_gpu(
    use_mixed_precision: bool = True,
    use_xla: bool = True,
    memory_growth: bool = True,
) -> bool:
    """Оптимальне налаштування GPU"""

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
    except Exception as error:
        _log_gpu_environment_diagnostics('list_physical_devices', error)
        _reset_to_cpu_policy(enable_logging=use_mixed_precision)
        return False

    if not gpus:
        _log_gpu_environment_diagnostics('TensorFlow не виявив сумісних GPU')
        _reset_to_cpu_policy(enable_logging=use_mixed_precision)
        return False

    try:
        tf.config.set_visible_devices(gpus, 'GPU')
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        if use_xla:
            try:
                tf.config.optimizer.set_jit(True)
                logger.info("✅ XLA JIT увімкнено")
            except Exception as jit_error:
                logger.warning("⚠️ Не вдалося увімкнути XLA JIT: %s", jit_error)

        if use_mixed_precision:
            from tensorflow import keras

            keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("✅ Mixed precision увімкнено")

        logger.info("✅ GPU доступний: %d пристроїв", len(gpus))
        return True
    except Exception as error:
        _log_gpu_environment_diagnostics('configure_gpu', error)
        _reset_to_cpu_policy(enable_logging=use_mixed_precision)
        return False


def get_gpu_info() -> dict:
    """Отримання інформації про GPU"""

    info: dict = {
        'available': False,
        'count': 0,
        'details': [],
    }

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            info['driver_version'] = driver_version

            device_count = pynvml.nvmlDeviceGetCount()
            info['count'] = device_count
            info['available'] = device_count > 0

            if hasattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2"):
                cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            elif hasattr(pynvml, "nvmlSystemGetCudaDriverVersion"):
                cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()
            else:
                cuda_driver_version = None

            if cuda_driver_version:
                info['cuda_driver_version'] = _format_cuda_version(cuda_driver_version)

            details = []
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                details.append({
                    'id': index,
                    'name': name,
                    'memory_used_mb': memory.used / (1024 ** 2),
                    'memory_total_mb': memory.total / (1024 ** 2),
                    'memory_percent': (memory.used / memory.total) * 100 if memory.total else 0,
                    'gpu_utilization': utilization.gpu,
                    'memory_utilization': utilization.memory,
                })

            info['details'] = details
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        info['error'] = 'pynvml_not_installed'
    except Exception as error:
        info['error'] = str(error)

    try:
        tf_gpus = tf.config.experimental.list_physical_devices('GPU')
        info['tensorflow_visible'] = len(tf_gpus)
    except Exception as error:
        info['tensorflow_error'] = str(error)

    return info