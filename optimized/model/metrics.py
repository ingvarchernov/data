"""
Custom Keras Metrics

Спеціалізовані метрики для моделі прогнозування цін:
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def mape(y_true, y_pred):
    """
    MAPE метрика з кращою стабільністю
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value (%)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + 1e-6))) * 100


@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    """
    Точність напрямку руху ціни (зростання/падіння)
    
    Args:
        y_true: True values (price changes)
        y_pred: Predicted values (price changes)
    
    Returns:
        Directional accuracy (0-1)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Рахуємо знак змін ціни
    true_direction = tf.sign(y_true + 1e-8)  # додаємо epsilon щоб уникнути 0
    pred_direction = tf.sign(y_pred + 1e-8)
    
    # Порівнюємо напрямки
    correct = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
    return tf.reduce_mean(correct)


@tf.keras.utils.register_keras_serializable()
def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Export all metrics
__all__ = [
    'mape',
    'directional_accuracy',
    'rmse',
]
