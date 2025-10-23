"""
Training Module - Модуль для тренування моделей

Містить:
- BaseModelTrainer - базовий клас для всіх trainers
- FeatureEngineer - розрахунок та обробка features
- DataLoader - завантаження даних з різних джерел
- Utilities - допоміжні функції (sequences, normalization)
- Models - специфічні реалізації (classification, enhanced, optimized)
"""

from .base_trainer import BaseModelTrainer
from .feature_engineering import FeatureEngineer
from .data_loader import DataLoader
from .utils import create_sequences, normalize_data, split_data

__all__ = [
    'BaseModelTrainer',
    'FeatureEngineer',
    'DataLoader',
    'create_sequences',
    'normalize_data',
    'split_data',
]
