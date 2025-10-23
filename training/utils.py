"""
Training Utilities - Спільні функції для тренування

Функції для:
- Створення sequences (для LSTM/RNN)
- Нормалізація даних
- Train/test/validation split
- Розрахунок class weights
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 60,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Створення sequences для LSTM/RNN моделей
    
    Args:
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        sequence_length: Довжина послідовності
        step: Крок між послідовностями
    
    Returns:
        X_seq: (n_sequences, sequence_length, n_features)
        y_seq: (n_sequences,)
    """
    X_sequences = []
    y_sequences = []
    
    for i in range(sequence_length, len(X), step):
        X_sequences.append(X[i-sequence_length:i])
        y_sequences.append(y[i])
    
    return np.array(X_sequences), np.array(y_sequences)


def normalize_data(
    X: np.ndarray,
    scaler_type: str = 'robust',
    scaler: Optional[any] = None,
    fit: bool = True
) -> Tuple[np.ndarray, any]:
    """
    Нормалізація даних
    
    Args:
        X: Дані для нормалізації
        scaler_type: Тип scaler ('robust' або 'standard')
        scaler: Існуючий scaler (якщо fit=False)
        fit: Чи треба fit scaler
    
    Returns:
        X_normalized: Нормалізовані дані
        scaler: Fitted scaler
    """
    if scaler is None:
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
    
    if fit:
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = scaler.transform(X)
    
    return X_normalized, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Розділення на train/val/test
    
    Args:
        X: Features
        y: Targets
        train_size: Розмір train set (0-1)
        val_size: Розмір validation set (0-1)
        test_size: Розмір test set (0-1)
        shuffle: Чи перемішувати дані
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, \
        "train_size + val_size + test_size must equal 1.0"
    
    n = len(X)
    
    if shuffle:
        indices = np.random.permutation(n)
        X = X[indices]
        y = y[indices]
    
    # Розрахунок індексів
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    # Розділення
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_class_weights(
    y: np.ndarray,
    clip_min: float = 0.5,
    clip_max: float = 3.0
) -> dict:
    """
    Розрахунок class weights для дисбалансу класів
    
    Args:
        y: Target array з класами
        clip_min: Мінімальна вага
        clip_max: Максимальна вага
    
    Returns:
        class_weights: Dict {class: weight}
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    # Clip weights
    weights = np.clip(weights, clip_min, clip_max)
    
    return {int(c): float(w) for c, w in zip(classes, weights)}


def create_classification_targets(
    returns: pd.Series,
    down_threshold: float = -0.005,
    up_threshold: float = 0.005
) -> np.ndarray:
    """
    Створення класифікаційних target'ів з returns
    
    Args:
        returns: Pandas Series з returns
        down_threshold: Поріг для DOWN класу
        up_threshold: Поріг для UP класу
    
    Returns:
        targets: Array з класами (0=DOWN, 1=NEUTRAL, 2=UP)
    """
    conditions = [
        returns < down_threshold,
        (returns >= down_threshold) & (returns <= up_threshold),
        returns > up_threshold
    ]
    
    targets = np.select(conditions, [0, 1, 2], default=1)
    return targets


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, ...]:
    """
    Temporal split (без shuffle) для time-series
    
    Args:
        X: Features
        y: Targets
        train_ratio: Розмір train set
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test
