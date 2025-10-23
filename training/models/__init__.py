"""
Training Models - Специфічні реалізації тренування

Містить готові trainers для різних підходів:
- OptimizedTrainer - тренування з відібраними features
- AdvancedTrainer - тренування з Rust індикаторами
"""

from .optimized_trainer import OptimizedTrainer
from .advanced_trainer import AdvancedTrainer

__all__ = [
    'OptimizedTrainer',
    'AdvancedTrainer',
]

__all__ = []
