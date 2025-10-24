"""
Backward compatibility wrapper for optimized model modules

This file provides backward compatibility for imports from 'optimized_model'.
New code should use: from optimized.model import ...
"""
import warnings

warnings.warn(
    "Importing from 'optimized_model' is deprecated. "
    "Use 'from optimized.model import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import all model components
from optimized.model.metrics import mape, directional_accuracy
from optimized.model.callbacks import (
    DatabaseHistoryCallback,
    DenormalizedMetricsCallback
)
from optimized.model.layers import TransformerBlock, PositionalEncoding

# Placeholder for DetailedLoggingCallback (if doesn't exist)
try:
    from optimized.model.callbacks import DetailedLoggingCallback
except ImportError:
    DetailedLoggingCallback = None

# Legacy class for backward compatibility
class OptimizedPricePredictionModel:
    """
    Legacy wrapper - use training.models.OptimizedTrainer instead
    
    This class is deprecated and kept only for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OptimizedPricePredictionModel is deprecated. "
            "Use 'from training.models import OptimizedTrainer' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Minimal implementation for compatibility
        self.config = kwargs

__all__ = [
    'OptimizedPricePredictionModel',
    'mape',
    'directional_accuracy',
    'DatabaseHistoryCallback',
    'DenormalizedMetricsCallback',
    'TransformerBlock',
    'PositionalEncoding',
]

if DetailedLoggingCallback is not None:
    __all__.append('DetailedLoggingCallback')
