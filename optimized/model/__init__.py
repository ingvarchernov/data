"""
Optimized Model Package

Модульна структура для моделі прогнозування цін:
- metrics: Custom Keras metrics (MAPE, Directional Accuracy)
- callbacks: Custom callbacks (DB history, denormalized metrics)
- layers: Custom layers (Transformer, Positional Encoding)
- architectures: Model architectures (будуть додані)
- predictor: Prediction utilities (будуть додані)

Backward compatibility:
- Імпортує основні компоненти для зворотної сумісності з optimized_model.py
"""

from .metrics import mape, directional_accuracy, rmse
from .callbacks import DatabaseHistoryCallback, DenormalizedMetricsCallback
from .layers import TransformerBlock, PositionalEncoding

__all__ = [
    # Metrics
    'mape',
    'directional_accuracy',
    'rmse',
    # Callbacks
    'DatabaseHistoryCallback',
    'DenormalizedMetricsCallback',
    # Layers
    'TransformerBlock',
    'PositionalEncoding',
]
